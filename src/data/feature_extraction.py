import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

def extract_tabular(df):
    """
    Extracts purely tabular data for the Level 1 Tree Models and Level 2 Meta-Model.
    Generates a 2223-dimensional feature space combining Morgan Fingerprints, 
    MACCS Keys, and explicit physicochemical descriptors.
    """
    X_fp, X_desc = [], []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    for smiles in df['SMILE']:
        mol = Chem.MolFromSmiles(smiles)
        # Edge Case Fallback: Prevent pipeline crashes on invalid SMILES
        if mol is None:
            X_fp.append(np.zeros(2048 + 167))
            X_desc.append(np.zeros(8))
            continue
            
        fp = list(mfpgen.GetFingerprint(mol))
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
        desc = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol), Descriptors.RingCount(mol)
        ]
        X_fp.append(fp + maccs)
        X_desc.append(desc)
        
    return np.nan_to_num(np.array(X_fp, dtype=np.float32)), np.nan_to_num(np.array(X_desc, dtype=np.float32))
    
def get_atom_features(atom):
    """
    Extracts rich 2D topological atom features for Message Passing Neural Networks.
    Returns 9 discrete features per atom.
    """
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    num_hs = atom.GetTotalNumHs()
    is_aromatic = int(atom.GetIsAromatic())
    is_in_ring = int(atom.IsInRing())
    
    hyb = atom.GetHybridization()
    hyb_sp = int(hyb == Chem.HybridizationType.SP)
    hyb_sp2 = int(hyb == Chem.HybridizationType.SP2)
    hyb_sp3 = int(hyb == Chem.HybridizationType.SP3)
    
    return [atomic_num, degree, formal_charge, num_hs, is_aromatic, is_in_ring, hyb_sp, hyb_sp2, hyb_sp3]

def get_bond_features(bond):
    """
    Generates one-hot encoded bond types and topological properties.
    Returns 6 discrete features per bond.
    """
    bt = bond.GetBondType()
    bt_single = int(bt == Chem.BondType.SINGLE)
    bt_double = int(bt == Chem.BondType.DOUBLE)
    bt_triple = int(bt == Chem.BondType.TRIPLE)
    bt_aromatic = int(bt == Chem.BondType.AROMATIC)
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())
    
    return [bt_single, bt_double, bt_triple, bt_aromatic, is_conjugated, is_in_ring]