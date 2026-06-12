import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import warnings

# Internal Imports
from src.data.feature_extraction import extract_tabular
from src.models.gnn_engines import GINModel, GATModel
from src.models.tabular_engines import get_tabular_models
from src.training.train_level1 import train_cv
from src.training.splitters import scaffold_split
from src.utils.metrics import find_best_thresholds
from src.utils.rank_transforms import to_rank_probs

warnings.filterwarnings('ignore')

def main():
    print("--- Multi-Modal Molecular Stacking Pipeline: Training Phase ---")
    df_train = pd.read_csv('data/raw/train.csv')
    y_train_multi = df_train[['P1', 'P2', 'P3', 'P4', 'P5']].values
    
    NUM_NODE_FEATURES = 9
    NUM_EDGE_FEATURES = 6

    # --- LEVEL 1: GRAPH MODELS ---
    print("1. Training GIN Models with Jumping Knowledge...")
    gin_models, gin_oof = train_cv(df_train, GINModel, NUM_NODE_FEATURES, NUM_EDGE_FEATURES, n_splits=8, epochs=30)
    
    print("2. Training GATv2 Models with Jumping Knowledge...")
    gat_models, gat_oof = train_cv(df_train, GATModel, NUM_NODE_FEATURES, NUM_EDGE_FEATURES, n_splits=8, epochs=30)

    # --- LEVEL 1: TABULAR MODELS ---
    print("3. Extracting 2223-Dimensional Tabular Matrices...")
    X_train_fp, X_train_desc = extract_tabular(df_train)
    X_train_tab = np.hstack((X_train_fp, X_train_desc))
    
    scaler = StandardScaler()
    X_train_tab_scaled = scaler.fit_transform(X_train_tab)
    joblib.dump(scaler, 'data/processed/tabular_scaler.pkl')

    print("4. Training Tabular Baselines (XGB, RF, & Deep MLP)...")
    spw = np.mean((len(y_train_multi) - y_train_multi.sum(axis=0)) / (y_train_multi.sum(axis=0) + 1e-5))
    xgb_oof, rf_oof, mlp_oof = np.zeros((len(df_train), 5)), np.zeros((len(df_train), 5)), np.zeros((len(df_train), 5))
    
    xgb, rf, mlp = get_tabular_models(scale_pos_weight=spw)
    splits = scaffold_split(df_train, n_splits=8) 
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Fit and extract probabilities
        xgb.fit(X_train_tab[train_idx], y_train_multi[train_idx])
        xgb_oof[val_idx] = np.column_stack([p[:, 1] for p in xgb.predict_proba(X_train_tab[val_idx])])
        
        rf.fit(X_train_tab[train_idx], y_train_multi[train_idx])
        rf_oof[val_idx] = np.column_stack([p[:, 1] for p in rf.predict_proba(X_train_tab[val_idx])])
        
        mlp.fit(X_train_tab_scaled[train_idx], y_train_multi[train_idx])
        mlp_oof[val_idx] = np.column_stack([p[:, 1] for p in mlp.predict_proba(X_train_tab_scaled[val_idx])])

    # --- LEVEL 2: META-MODEL STACKING ---
    print("5. Synthesizing OOF Predictions for Meta-Orchestrator...")
    
    # Apply rank transformations to stabilize the meta-model
    X_train_meta = np.hstack((
        to_rank_probs(gin_oof), 
        to_rank_probs(gat_oof), 
        to_rank_probs(xgb_oof), 
        to_rank_probs(rf_oof), 
        to_rank_probs(mlp_oof), 
        X_train_desc
    ))
    
    meta_models, oof_meta_preds = [], np.zeros((len(df_train), 5))
    for i in range(5): 
        meta_lr = LogisticRegression(max_iter=1500, C=0.05, class_weight='balanced')
        meta_lr.fit(X_train_meta, y_train_multi[:, i])
        oof_meta_preds[:, i] = meta_lr.predict_proba(X_train_meta)[:, 1]
        meta_models.append(meta_lr)

    print("6. Optimizing Dynamic MCC Thresholds...")
    optimal_thresholds = find_best_thresholds(y_train_multi, oof_meta_preds)
    print(f"Optimal Clamped Thresholds (P1-P5): {optimal_thresholds}")
    
    # In a fully deployed setup, you would save all models to disk here.
    # For repository completeness, we save the critical thresholds.
    joblib.dump(optimal_thresholds, 'data/processed/optimal_thresholds.pkl')
    print("Training Complete. Ready for Inference.")

if __name__ == "__main__":
    main()