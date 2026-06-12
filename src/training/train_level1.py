import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

# Import internal modules
from src.data.dataset import MoleculeDataset
from src.training.loss import FocalLoss
from src.training.splitters import scaffold_split

def predict(model, loader, device):
    """
    Generates probability predictions from a trained PyTorch model.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs = torch.sigmoid(out).cpu().numpy()
            
            # Ensure proper shape for single-item batches
            if probs.ndim == 1: 
                probs = np.expand_dims(probs, axis=0)
            preds.extend(probs.tolist())
            
    return np.array(preds)

def train_cv(df_train, model_class, num_node_features, num_edge_features, n_splits=8, epochs=30):
    """
    Cross-validation training loop for Level-1 Graph Neural Networks.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_train_full = df_train[['P1', 'P2', 'P3', 'P4', 'P5']].values
    
    # Calculate dynamic positive weights for the Focal Loss based on dataset imbalance
    num_positives = y_train_full.sum(axis=0)
    num_negatives = len(y_train_full) - num_positives
    pos_weights = torch.tensor(num_negatives / (num_positives + 1e-5), dtype=torch.float).to(device)
    
    oof_preds = np.zeros((len(df_train), 5))
    models = []
    
    # Utilize our custom chemical scaffold splitter
    splits = scaffold_split(df_train, n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(splits):
        # IMPORTANT: drop_last=True prevents fatal PyTorch errors when the final training batch 
        # contains only 1 molecule, which mathematically breaks standard deviation in BatchNorm1d.
        train_loader = DataLoader(
            MoleculeDataset(df_train.iloc[train_idx]), 
            batch_size=64, 
            shuffle=True, 
            drop_last=True
        )
        val_loader = DataLoader(
            MoleculeDataset(df_train.iloc[val_idx]), 
            batch_size=64, 
            shuffle=False
        )
        
        # Initialize model, optimizer, scheduler, and custom loss
        model = model_class(num_node_features, num_edge_features).to(device)
        optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = FocalLoss(gamma=2.0, pos_weight=pos_weights)
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                
                # Forward pass and loss calculation
                loss = criterion(model(data), data.y) 
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
        # Generate Out-Of-Fold (OOF) predictions for Level-2 Meta-Model training
        oof_preds[val_idx] = predict(model, val_loader, device)
        models.append(model)
        
    return models, oof_preds