import numpy as np
from sklearn.metrics import matthews_corrcoef

def find_best_thresholds(y_true, y_pred_probs):
    """
    Finds the optimal classification threshold for each target to maximize the 
    Matthews Correlation Coefficient (MCC). 
    
    Includes an engineered search space clamp (0.30 to 0.90) to prevent the dynamic 
    threshold optimizer from aggressively chasing perfect precision and tanking 
    the overall MCC score through excessive False Negatives.
    """
    best_thresholds = []
    
    # Iterate over all 5 binary targets
    for i in range(5):
        best_thresh = 0.5
        best_mcc = -1
        
        # Iterate through potential thresholds
        for t in np.arange(0.30, 0.91, 0.02):
            preds = (y_pred_probs[:, i] > t).astype(int)
            mcc = matthews_corrcoef(y_true[:, i], preds)
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t
                
        best_thresholds.append(best_thresh)
        
    return best_thresholds

def calculate_mean_mcc(y_true, y_pred_probs, thresholds):
    """
    Calculates the macro-averaged MCC across all 5 molecular properties.
    """
    mcc_scores = []
    for i in range(5):
        preds = (y_pred_probs[:, i] > thresholds[i]).astype(int)
        mcc_scores.append(matthews_corrcoef(y_true[:, i], preds))
    return np.mean(mcc_scores)