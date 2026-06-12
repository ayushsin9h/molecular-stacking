import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINEConv, GATv2Conv, global_mean_pool

class GINModel(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN) with Jumping Knowledge and Global Feature Fusion.
    Optimized for multi-label binary property classification.
    """
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128, fp_dim=2223):
        super().__init__()
        # 3 Layers of Message Passing with sequential MLPs
        nn1 = Sequential(Linear(num_node_features, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)
        
        nn2 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(nn2, edge_dim=num_edge_features)
        
        nn3 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(nn3, edge_dim=num_edge_features)
        
        # FC accepts: (3 layers * hidden_dim for Jumping Knowledge) + fp_dim
        jk_dim = hidden_dim * 3
        self.fc1 = Linear(jk_dim + fp_dim, hidden_dim * 2)
        self.drop = torch.nn.Dropout(0.5)
        self.fc2 = Linear(hidden_dim * 2, 5) # 5 Binary Targets (P1-P5)

    def forward(self, data):
        x, edge_index, edge_attr, batch, fp = data.x, data.edge_index, data.edge_attr, data.batch, data.fp
        
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        
        # JUMPING KNOWLEDGE: Concatenate all layer representations to prevent over-smoothing
        x_jk = torch.cat([x1, x2, x3], dim=-1)
        
        # Global Mean Pooling
        x_pool = global_mean_pool(x_jk, batch)
        
        # MULTI-MODAL FUSION: Combine pooled graph representation with Global Fingerprints
        x_out = torch.cat([x_pool, fp.squeeze(1)], dim=1)
        
        x_out = self.drop(F.relu(self.fc1(x_out)))
        return self.fc2(x_out)


class GATModel(torch.nn.Module):
    """
    Graph Attention Network v2 (GATv2) with Jumping Knowledge and Global Feature Fusion.
    Allows the model to learn the relative importance of neighboring atoms and bonds.
    """
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128, heads=4, fp_dim=2223):
        super().__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_dim, heads=heads, edge_dim=num_edge_features, concat=False)
        self.bn1 = BatchNorm1d(hidden_dim)
        
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, edge_dim=num_edge_features, concat=False)
        self.bn2 = BatchNorm1d(hidden_dim)
        
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, edge_dim=num_edge_features, concat=False)
        self.bn3 = BatchNorm1d(hidden_dim)
        
        jk_dim = hidden_dim * 3
        self.fc1 = Linear(jk_dim + fp_dim, hidden_dim * 2)
        self.drop = torch.nn.Dropout(0.5)
        self.fc2 = Linear(hidden_dim * 2, 5)

    def forward(self, data):
        x, edge_index, edge_attr, batch, fp = data.x, data.edge_index, data.edge_attr, data.batch, data.fp
        
        x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index, edge_attr)))
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index, edge_attr)))
        
        # JUMPING KNOWLEDGE
        x_jk = torch.cat([x1, x2, x3], dim=-1)
        
        # Global Mean Pooling
        x_pool = global_mean_pool(x_jk, batch)
        
        # MULTI-MODAL FUSION
        x_out = torch.cat([x_pool, fp.squeeze(1)], dim=1)
        x_out = self.drop(F.relu(self.fc1(x_out)))
        
        return self.fc2(x_out)