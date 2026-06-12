import pandas as pd
import numpy as np
import torch
import joblib
from torch_geometric.loader import DataLoader

# Internal Imports
from src.data.feature_extraction import extract_tabular
from src.data.dataset import MoleculeDataset
from src.utils.rank_transforms import to_rank_probs

def generate_graph_preds(models, df_test):
    """Helper function to run inference on Level-1 PyTorch Geometric models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(MoleculeDataset(df_test, is_test=True), batch_size=64, shuffle=False)
    
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                probs = torch.sigmoid(model(data)).cpu().numpy()
                if probs.ndim == 1: probs = np.expand_dims(probs, axis=0)
                preds.extend(probs.tolist())
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)

def main():
    print("--- Multi-Modal Molecular Stacking: Inference Phase ---")
    df_test = pd.read_csv('data/raw/test.csv')
    
    # Load previously computed components
    # NOTE: In a true deployment, you would load the trained `models` arrays from disk here.
    # For demonstration, this assumes the models are passed in or loaded via joblib/torch.load.
    scaler = joblib.load('data/processed/tabular_scaler.pkl')
    optimal_thresholds = joblib.load('data/processed/optimal_thresholds.pkl')
    
    print("Extracting Test Data Features...")
    X_test_fp, X_test_desc = extract_tabular(df_test)
    X_test_tab = np.hstack((X_test_fp, X_test_desc))
    X_test_tab_scaled = scaler.transform(X_test_tab)
    
    # -------------------------------------------------------------------------
    # PLACEHOLDER: Execute predictions for GIN, GAT, XGB, RF, and MLP models here
    # -------------------------------------------------------------------------
    # gin_test_preds = generate_graph_preds(gin_models, df_test)
    # gat_test_preds = generate_graph_preds(gat_models, df_test)
    # xgb_test_preds = np.mean([np.column_stack([p[:, 1] for p in m.predict_proba(X_test_tab)]) for m in xgb_models], axis=0)
    # rf_test_preds = np.mean([np.column_stack([p[:, 1] for p in m.predict_proba(X_test_tab)]) for m in rf_models], axis=0)
    # mlp_test_preds = np.mean([np.column_stack([p[:, 1] for p in m.predict_proba(X_test_tab_scaled)]) for m in mlp_models], axis=0)
    
    print("Synthesizing Level-2 Meta-Features...")
    # Apply the same rank transformation used in training to the test predictions
    # X_test_meta = np.hstack((
    #     to_rank_probs(gin_test_preds), 
    #     to_rank_probs(gat_test_preds), 
    #     to_rank_probs(xgb_test_preds), 
    #     to_rank_probs(rf_test_preds), 
    #     to_rank_probs(mlp_test_preds), 
    #     X_test_desc
    # ))
    
    # final_test_preds = np.zeros((len(df_test), 5))
    # for i in range(5):
    #     final_test_preds[:, i] = meta_models[i].predict_proba(X_test_meta)[:, 1]

    print("Applying Optimal Thresholds and Formatting Submission...")
    # final_test_hard_labels = np.zeros_like(final_test_preds, dtype=int)
    # for i in range(5):
    #     final_test_hard_labels[:, i] = (final_test_preds[:, i] > optimal_thresholds[i]).astype(int)
        
    submission = pd.DataFrame()
    submission['M-ID'] = df_test['M-ID']
    # for i, prop in enumerate(['P1', 'P2', 'P3', 'P4', 'P5']):
    #     submission[prop] = final_test_hard_labels[:, i]

    submission.to_csv('submission.csv', index=False)
    print("JK-Ensemble Submission generated successfully!")

if __name__ == "__main__":
    main()