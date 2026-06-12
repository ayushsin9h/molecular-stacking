import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles):
    """
    Extracts the core 2D ring structure (Murcko Scaffold) of a molecule.
    Contains explicit safety fallbacks to prevent pipeline crashes on bad SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return ""
    
    # Explicitly strip all stereochemistry to prevent "bad bond stereo" crashes
    Chem.RemoveStereochemistry(mol)
    
    try:
        # Try-except block ensures a single bad molecule won't crash the whole pipeline
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception:
        # Fallback: If RDKit still fails, return an empty string to group it with other edge cases
        return ""

def scaffold_split(df, n_splits=5):
    """
    Splits the dataset based on core 2D ring structures rather than random sampling.
    Perfectly simulates the hidden Private Leaderboard by forcing models to 
    generalize to unseen chemical backbones.
    """
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(df['SMILE']):
        scaffolds[generate_scaffold(smiles)].append(i)
        
    # Sort scaffolds from largest to smallest
    scaffold_sets = [idx_list for _, idx_list in sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)]
    
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    
    # Distribute scaffolds to maintain balanced fold sizes
    for scaffold_set in scaffold_sets:
        smallest_fold_idx = np.argmin(fold_sizes)
        folds[smallest_fold_idx].extend(scaffold_set)
        fold_sizes[smallest_fold_idx] += len(scaffold_set)
        
    all_indices = np.arange(len(df))
    # Return a list of tuples: (train_indices, validation_indices)
    return [(np.setdiff1d(all_indices, val_idx), val_idx) for val_idx in folds]