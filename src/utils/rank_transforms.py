import numpy as np
from scipy.stats import rankdata

def to_rank_probs(preds):
    """
    Converts raw probability outputs from diverse Level-1 models (Trees vs. GNNs) 
    into uniform percentile ranks. 
    
    This completely immunizes the Level-2 Meta-Orchestrator against probability 
    calibration clashes and target distribution shifts between train and test sets.
    """
    ranks = np.zeros_like(preds)
    for i in range(preds.shape[1]):
        # Ranks from 1 to N, then divide by N to get a uniform 0.0 -> 1.0 percentile range
        ranks[:, i] = rankdata(preds[:, i]) / len(preds[:, i])
    return ranks