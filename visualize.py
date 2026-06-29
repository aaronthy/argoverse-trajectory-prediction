import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from model import TrajectoryPredictor
from train import ArgoverseDataset
import subprocess

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load val data
obs = np.load('/mnt/d/argoverse_data/val_obs.npy')
future = np.load('/mnt/d/argoverse_data/val_future.npy')
others = np.load('/mnt/d/argoverse_data/val_others.npy')

# Load model
model_name = 'model_large'
batch_size = 64
model = TrajectoryPredictor(hidden_dim=128, num_heads=8, num_layers=4)
pth_files = glob.glob(f'{model_name}_bs{batch_size}_epoch*.pth')
latest = max(pth_files, key=os.path.getctime)
print(f"Loading: {latest}")
model.load_state_dict(torch.load(latest))
model = model.to(device)
model.eval()

edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10],
                            [1,2,3,4,5,6,7,8,9,10,0]], dtype=torch.long).to(device)

# Pick a random sequence
idx = np.random.randint(0, len(obs))
batch_obs = torch.FloatTensor(obs[idx:idx+1]).to(device)
batch_others = torch.FloatTensor(others[idx:idx+1]).to(device)
batch_future = torch.FloatTensor(future[idx:idx+1]).to(device)

# Predict
with torch.no_grad():
    pred = model(batch_obs, batch_others, edge_index)

# Convert to numpy
obs_np = obs[idx]
future_np = future[idx]
pred_np = pred[0].cpu().numpy()

# Plot
plt.figure(figsize=(10, 8))
plt.plot(obs_np[:, 0], obs_np[:, 1], 'b-o', label='Observed (past)', markersize=4)
plt.plot(future_np[:, 0], future_np[:, 1], 'g-o', label='Real future', markersize=4)
plt.plot(pred_np[:, 0], pred_np[:, 1], 'r-o', label='Predicted future', markersize=4)
plt.plot(obs_np[0, 0], obs_np[0, 1], 'bs', markersize=10, label='Start')
plt.plot(future_np[-1, 0], future_np[-1, 1], 'g*', markersize=10, label='Real end')
plt.plot(pred_np[-1, 0], pred_np[-1, 1], 'r*', markersize=10, label='Predicted end')
plt.legend()
plt.title(f'Trajectory Prediction - Sequence {idx}')
plt.xlabel('X (normalized)')
plt.ylabel('Y (normalized)')
plt.savefig(f'prediction_seq{idx}.png')
print(f"Saved prediction_seq{idx}.png")
subprocess.run(['code', f'prediction_seq{idx}.png'])