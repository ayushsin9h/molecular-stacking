from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_tabular_models(scale_pos_weight):
    """
    Initializes and returns the Level-1 tabular base estimators wrapped 
    for multi-output classification. 
    
    Parameters:
    -----------
    scale_pos_weight : float
        The global imbalance weight calculated from the training set to fix tree bias
        towards the negative (majority) class.
        
    Returns:
    --------
    tuple : (MultiOutputClassifier, MultiOutputClassifier, MultiOutputClassifier)
        The un-fitted XGBoost, Random Forest, and Deep MLP models.
    """
    
    # 1. Balanced XGBoost
    xgb = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=400, 
            learning_rate=0.02, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            scale_pos_weight=scale_pos_weight, 
            n_jobs=-1, 
            random_state=42
        )
    )
    
    # 2. Balanced Random Forest
    rf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=300, 
            max_depth=15, 
            min_samples_split=5, 
            class_weight='balanced', 
            n_jobs=-1, 
            random_state=42
        )
    )
    
    # 3. Deep Tabular MLP
    # Three hidden layers to capture deep chemical interactions from the dense fingerprints
    mlp = MultiOutputClassifier(
        MLPClassifier(
            hidden_layer_sizes=(512, 256, 64), 
            activation='relu', 
            max_iter=200, 
            alpha=1e-3, 
            early_stopping=True, 
            random_state=42
        )
    )
    
    return xgb, rf, mlp