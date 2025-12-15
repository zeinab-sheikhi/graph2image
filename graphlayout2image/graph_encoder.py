import enum
import torch 
import torch.nn as nn 
from torch_geometric.nn import GCNConv, global_mean_pool 
from torch_geometric.data import Data, Batch


class ScenceGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        
        self.shape_embed = nn.Embedding(3, 32)
        self.color_embed = nn.Embedding(4, 32)

        self.conv1 = GCNConv(64, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, batch_graphs: Batch):

        data_list = []
        for graph in batch_graphs:
            shape_ids = torch.tensor([self.shape_to_id(n["shape"]) for n in graph["nodes"]]) 
            color_ids = torch.tensor([self.color_to_id(n["color"]) for n in graph["nodes"]])

            shape_emb = self.shape_embed(shape_ids)
            color_emb = self.color_embed(color_ids)

            x = torch.cat([shape_emb, color_emb], dim=-1)

            num_nodes = len(graph["nodes"])
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
            data_list.append(Data(x=x, edge_index=edge_index))
        
        batch = Batch.from_data_list(data_list)

        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = self.proj(x)

        node_features = []
        ptr = 0
        for data in data_list:
            num_nodes = data.num_nodes
            node_features.append(x[ptr: ptr + num_nodes])
            ptr += num_nodes
        
        max_nodes = max(nf.shape[0] for nf in node_features)
        padded = torch.zeros(len(node_features), max_nodes, x.shape[-1])
        for i, nf in enumerate(node_features):
            padded[i: nf.shape[0]] = nf 
        
        return padded
    
    def shape_to_id(self, shape: str):
        return {'circle': 0, 'square': 1, 'triangle': 2}[shape]
    
    def color_to_id(self, color: str):
        return {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3}[color]
