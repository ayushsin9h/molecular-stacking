import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """
    Custom Focal Loss to handle extreme target imbalance.
    Dynamically scales the penalty for missing rare active molecules, forcing 
    the network to focus on the minority class to maximize the MCC metric.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # reduction='none' allows us to apply the focal weight element-wise
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        # Apply the focal loss equation
        focal_loss = self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce
        
        return focal_loss.mean()