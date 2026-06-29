import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Trajectory Prediction Demo")
st.write("GNN + Transformer based driving trajectory prediction on Argoverse dataset")

@st.cache_data
def load_data():
    obs = np.load('sample_obs.npy')
    future = np.load('sample_future.npy')
    predictions = np.load('sample_predictions.npy')
    return obs, future, predictions

obs, future, predictions = load_data()

idx = st.slider("Select sequence", 0, len(obs)-1, 0)

obs_np = obs[idx]
future_np = future[idx]
pred_np = predictions[idx]

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

ade = np.mean(np.sqrt(np.sum((pred_np - future_np)**2, axis=-1)))
fde = np.sqrt(np.sum((pred_np[-1] - future_np[-1])**2))
st.metric("ADE (this sequence)", f"{ade:.4f} m")
st.metric("FDE (this sequence)", f"{fde:.4f} m")