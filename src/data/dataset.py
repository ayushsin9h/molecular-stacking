import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

# Import internal feature extraction functions
from .feature_extraction import get_atom_features, get_bond_features

def smiles_to_graph(smiles, labels=None):
    """
    Converts a single SMILES string into a PyTorch Geometric Data object containing
    both 2D graph topology (nodes/edges) and global tabular features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        mol = Chem.MolFromSmiles('C') # Dummy Atom Fallback silently handles broken SMILES
    
    # Process Nodes
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Process Edges
    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Message passing requires undirected edges (bidirectional)
        edge_indices += [[i, j], [j, i]]
        bond_feat = get_bond_features(bond)
        edge_features += [bond_feat, bond_feat]
        
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, 6), dtype=torch.float)
    
    # Generate Global Features (2223 Dimensions)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = list(mfpgen.GetFingerprint(mol))
    maccs = list(MACCSkeys.GenMACCSKeys(mol))
    desc = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol), Descriptors.RingCount(mol)
    ]
    
    global_features = torch.tensor(fp + maccs + desc, dtype=torch.float).unsqueeze(0)
    global_features = torch.nan_to_num(global_features, nan=0.0)
    
    # Process Labels for Multi-Label Binary Classification
    y = torch.tensor([labels], dtype=torch.float) if labels is not None else None
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, fp=global_features)

class MoleculeDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for Multi-Modal Molecular Stacking.
    Directly interfaces with Pandas DataFrames matching the Kaggle schema.
    """
    def __init__(self, dataframe, is_test=False):
        super().__init__()
        self.graphs = []
        for _, row in dataframe.iterrows():
            # If test set, labels are None. If train set, extract P1 through P5.
            labels = None if is_test else row[['P1', 'P2', 'P3', 'P4', 'P5']].values.astype(float)
            graph = smiles_to_graph(row['SMILE'], labels) 
            if graph is not None: 
                self.graphs.append(graph)

    def len(self): 
        return len(self.graphs)
        
    def get(self, idx): 
        return self.graphs[idx]