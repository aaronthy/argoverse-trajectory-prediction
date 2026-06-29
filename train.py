import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from model import TrajectoryPredictor
from tqdm import tqdm

# Dataset class
class ArgoverseDataset(Dataset):
    def __init__(self, obs, future, others):
        self.obs = torch.FloatTensor(obs)
        self.future = torch.FloatTensor(future)
        self.others = torch.FloatTensor(others)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.future[idx], self.others[idx]
    

if __name__ == '__main__':
    # Load data
    print("Loading data...")
    obs = np.load('/mnt/d/argoverse_data/train_obs.npy')
    future = np.load('/mnt/d/argoverse_data/train_future.npy')
    others = np.load('/mnt/d/argoverse_data/train_others.npy')

    batch_size = 64
    model_name = 'model_large'
    dataset = ArgoverseDataset(obs, future, others)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10,0]], dtype=torch.long)

    model = TrajectoryPredictor(hidden_dim=128, num_heads=8, num_layers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    edge_index = edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # lower lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      
    patience=5,      
    factor=0.5,   
    )
    criterion = nn.MSELoss()

    # Checkpoint resuming
    start_epoch = 100
    num_epochs = 150

    if start_epoch > 0:
        model.load_state_dict(torch.load(f'{model_name}_bs{batch_size}_epoch{start_epoch}.pth'))
        print(f"Resuming from epoch {start_epoch}")

    print("Starting training...")

    with mlflow.start_run():
    # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_dim", 128)
        mlflow.log_param("num_heads", 8)
        mlflow.log_param("num_layers", 4)
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("model_name", model_name)

        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            for batch_obs, batch_future, batch_others in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                batch_obs = batch_obs.to(device)
                batch_future = batch_future.to(device)
                batch_others = batch_others.to(device)
                pred = model(batch_obs, batch_others, edge_index)
                loss = criterion(pred, batch_future)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("loss", avg_loss, step=epoch+1)
            scheduler.step(avg_loss)
            torch.save(model.state_dict(), f'{model_name}_bs{batch_size}_epoch{epoch+1}.pth')

        print("Training done!")
