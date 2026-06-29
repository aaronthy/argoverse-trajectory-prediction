from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np

loader = ArgoverseForecastingLoader('/mnt/d/argoverse_data/train/data')
seq = loader[0]
df = seq.seq_df

agent_traj = seq.agent_traj
origin = agent_traj[19]
agent_traj_normalized = agent_traj - origin

# extract surrounding agents
others = []
for track_id in seq.track_id_list:
    track = df[df['TRACK_ID'] == track_id]
    obj_type = track['OBJECT_TYPE'].values[0]
    if obj_type == 'OTHERS':
        xy = track[['X','Y']].values
        xy_normalized = xy - origin
        others.append(xy_normalized)

print("Number of surrounding agents:", len(others))
for i, o in enumerate(others):
    print(f"Agent {i} shape: {o.shape}")

# Pad all agents to 50 timesteps
max_len = 50
padded_others = []
for o in others:
    pad_len = max_len - len(o)
    if pad_len > 0:
        padding = np.zeros((pad_len, 2))
        o = np.concatenate([padding, o], axis = 0)
    padded_others.append(o)

padded_others = np.array(padded_others)
print("Padded others shape:", padded_others.shape)

# Keep only N closest agents to the AGENT
N = 10
agent_last_pos = agent_traj[19] # last observed position of AGENT

# Calculate distance from each OTHER to AGENT
distances = []
for o in padded_others:
    last_pos = o[-1] #last position of this agent
    dist = np.linalg.norm(last_pos - agent_last_pos)
    distances.append(dist)

distances = np.array(distances)
closest_idx = np.argsort(distances)[:N]
closest_others = padded_others[closest_idx]

# If less tha N agents, pad with zeros
if len(closest_others) < N:
    pad = np.zeros((N - len(closest_others), 50, 2))
    closest_others = np.concatenate([closest_others, pad], axis = 0)

print("Closest others shape:", closest_others.shape)