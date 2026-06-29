import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from model import TrajectoryPredictor

st.title("Trajectory Prediction Demo")
st.write("GNN + Transformer based driving trajectory prediction on Argoverse dataset")

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrajectoryPredictor(hidden_dim=128, num_heads=8, num_layers=4)
    pth_files = glob.glob('model_large_bs64_epoch*.pth')
    latest = max(pth_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Load data
@st.cache_data
def load_data():
    obs = np.load('sample_obs.npy')
    future = np.load('sample_future.npy')
    others = np.load('sample_others.npy')
    return obs, future, others

model, device = load_model()
obs, future, others = load_data()

edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10],
                            [1,2,3,4,5,6,7,8,9,10,0]], dtype=torch.long).to(device)

# Sequence selector
idx = st.slider("Select sequence", 0, 99, 0)

# Predict
batch_obs = torch.FloatTensor(obs[idx:idx+1]).to(device)
batch_others = torch.FloatTensor(others[idx:idx+1]).to(device)

with torch.no_grad():
    pred = model(batch_obs, batch_others, edge_index)

obs_np = obs[idx]
future_np = future[idx]
pred_np = pred[0].cpu().numpy()

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(obs_np[:, 0], obs_np[:, 1], 'b-o', label='Observed (past)', markersize=4)
ax.plot(future_np[:, 0], future_np[:, 1], 'g-o', label='Real future', markersize=4)
ax.plot(pred_np[:, 0], pred_np[:, 1], 'r-o', label='Predicted future', markersize=4)
ax.plot(obs_np[0, 0], obs_np[0, 1], 'bs', markersize=10, label='Start')
ax.plot(future_np[-1, 0], future_np[-1, 1], 'g*', markersize=10, label='Real end')
ax.plot(pred_np[-1, 0], pred_np[-1, 1], 'r*', markersize=10, label='Predicted end')
ax.legend()
ax.set_title(f'Trajectory Prediction - Sequence {idx}')
ax.set_xlabel('X (normalized)')
ax.set_ylabel('Y (normalized)')

st.pyplot(fig)

# Show metrics
ade = np.mean(np.sqrt(np.sum((pred_np - future_np)**2, axis=-1)))
fde = np.sqrt(np.sum((pred_np[-1] - future_np[-1])**2))
st.metric("ADE (this sequence)", f"{ade:.4f} m")
st.metric("FDE (this sequence)", f"{fde:.4f} m")