import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 64):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        return x
    
# Test GNN
gnn = GNNEncoder(input_dim=2, hidden_dim=64)

# Fake data - 11 agents ( 1 Agent + 10 Others), each with x,y position
x = torch.randn(11,2)

# Fake edges - connect every agent to every other agent
edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10]], dtype = torch.long)

output = gnn(x, edge_index)
print("GNN output shape", output.shape)

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, num_heads=4, num_layers=2):
        super(TransformerPredictor, self).__init__()
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim * 30)

    def forward(self, obs, gnn_context):
        # obs: (batch, 20, 2)
        # gnn_context: (batch, hidden_dim)
        gnn_context = gnn_context.unsqueeze(1).repeat(1, 20, 1)
        x = torch.cat([obs, gnn_context], dim=-1)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.output_proj(x)
        x = x.view(-1, 30, 2)
        return x

# Test Transformer
transformer = TransformerPredictor(input_dim=2, hidden_dim=64, output_dim=2)

# Fake data
obs = torch.randn(4, 20, 2)        # batch of 4, 20 timesteps, x,y
gnn_context = torch.randn(4, 64)   # batch of 4, 64 GNN features

output = transformer(obs, gnn_context)
print("Transformer output shape:", output.shape)  # should be (4, 30, 2)

#Combine GNN and Transformer
class TrajectoryPredictor(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, num_layers=2):
        super(TrajectoryPredictor, self).__init__()
        self.gnn = GNNEncoder(input_dim=2, hidden_dim=hidden_dim)
        self.transformer = TransformerPredictor(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_heads=num_heads, num_layers=num_layers)

    def forward(self, obs, others, edge_index):
        # obs: (batch, 20, 2)
        # others: (batch, 10, 50, 2)
        # edge_index: (2, num_edges)
        
        batch_size = obs.shape[0]
        
        # Take last observed position of each other agent
        others_last = others[:, :, 19, :]  # (batch, 10, 2)
        
        # Combine AGENT last position with others
        agent_last = obs[:, 19, :].unsqueeze(1)  # (batch, 1, 2)
        all_agents = torch.cat([agent_last, others_last], dim=1)  # (batch, 11, 2)
        
        # Run GNN on each scene
        gnn_out = []
        for i in range(batch_size):
            out = self.gnn(all_agents[i].to(edge_index.device), edge_index)
            gnn_out.append(out[0])  # take AGENT's features (index 0)
        gnn_context = torch.stack(gnn_out)  # (batch, 64)
        
        # Run Transformer
        pred = self.transformer(obs.to(edge_index.device), gnn_context)
        return pred
    
# Test full model
model = TrajectoryPredictor(hidden_dim=64, num_heads=4, num_layers=2)

# Fake data
obs = torch.randn(4, 20, 2)           # 4 videos, 20 timesteps, x,y
others = torch.randn(4, 10, 50, 2)    # 4 videos, 10 agents, 50 timesteps, x,y
edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10,0]], dtype=torch.long)

pred = model(obs, others, edge_index)
print("Final prediction shape:", pred.shape)  # should be (4, 30, 2)