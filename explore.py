from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import matplotlib.pyplot as plt
import numpy as np

loader = ArgoverseForecastingLoader('/mnt/d/argoverse_data/train/data')
seq = loader[0]

df = seq.seq_df
agent_traj = seq.agent_traj

obs = agent_traj[:20]
future = agent_traj[20:]

plt.figure(figsize=(10, 8))

# Plot other agents
for track_id in seq.track_id_list:
    track = df[df['TRACK_ID'] == track_id]
    obj_type = track['OBJECT_TYPE'].values[0]
    if obj_type == 'OTHERS':
        plt.plot(track['X'].values, track['Y'].values, 'gray', alpha=0.5)
    elif obj_type == 'AV':
        plt.plot(track['X'].values, track['Y'].values, 'r--', label='AV')

plt.plot(obs[:, 0], obs[:, 1], 'b-o', label='Observed (past)')
plt.plot(future[:, 0], future[:, 1], 'g-o', label='Future (ground truth)')
plt.legend()
plt.title('Full Scene - Sequence 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('scene.png')
print("Saved to scene.png")