import torch
import numpy as np
from torch.utils.data import DataLoader
from model import TrajectoryPredictor
from train import ArgoverseDataset
import os
import glob
import mlflow

# Load val data
print("Loading val data...")
obs = np.load('/mnt/d/argoverse_data/val_obs.npy')
future = np.load('/mnt/d/argoverse_data/val_future.npy')
others = np.load('/mnt/d/argoverse_data/val_others.npy')

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ArgoverseDataset(obs, future, others)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10,0]], dtype=torch.long).to(device)

# Load latest saved model automatically
model_name = 'model_large'
batch_size = 64
model = TrajectoryPredictor(hidden_dim=128, num_heads=8, num_layers=4)  # match new architecture!
pth_files = glob.glob(f'{model_name}_bs{batch_size}_epoch*.pth')
latest = max(pth_files, key=os.path.getctime)
print(f"Loading: {latest}")
model.load_state_dict(torch.load(latest))
model = model.to(device)
model.eval()

# Evaluate
total_ade = 0
total_fde = 0

with torch.no_grad():
    for batch_obs, batch_future, batch_others in dataloader:
        batch_obs = batch_obs.to(device)
        batch_future = batch_future.to(device)
        batch_others = batch_others.to(device)

        pred = model(batch_obs, batch_others, edge_index)

        # ADE
        ade = torch.mean(torch.sqrt(torch.sum((pred - batch_future)**2, dim=-1)))
        # FDE
        fde = torch.mean(torch.sqrt(torch.sum((pred[:, -1, :] - batch_future[:, -1, :])**2, dim=-1)))

        total_ade += ade.item()
        total_fde += fde.item()

print(f"ADE: {total_ade/len(dataloader):.4f}")
print(f"FDE: {total_fde/len(dataloader):.4f}")

mlflow.set_experiment("trajectory_prediction")

with mlflow.start_run():
    mlflow.log_metric("ADE", total_ade/len(dataloader))
    mlflow.log_metric("FDE", total_fde/len(dataloader))
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("epochs", 150)
    print(f"ADE: {total_ade/len(dataloader):.4f}")
    print(f"FDE: {total_fde/len(dataloader):.4f}")
    print("Logged to MLflow!")