from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np

def preprocess_sequence(seq, N = 10):
    df = seq.seq_df
    agent_traj = seq.agent_traj

    # Normalize
    origin = agent_traj[19]
    agent_traj_normalized = agent_traj - origin

    #split obs and future
    obs = agent_traj_normalized[:20]
    future = agent_traj_normalized[20:]

    # Extract and normalize others
    others = []
    for track_id in seq.track_id_list:
        track = df[df['TRACK_ID'] == track_id]
        obj_type = track['OBJECT_TYPE'].values[0]
        if obj_type == 'OTHERS':
            xy = track[['X','Y']].values
            xy_normalized = xy - origin
            others.append(xy_normalized)

 # Pad others to 50 timesteps
    padded_others = []
    for o in others:
        pad_len = 50 - len(o)
        if pad_len > 0:
            padding = np.zeros((pad_len, 2))
            o = np.concatenate([padding, o], axis=0)
        padded_others.append(o)
    padded_others = np.array(padded_others) if padded_others else np.zeros((0, 50, 2))

 # Keep N closest agents
    if len(padded_others) > 0:
        agent_last_pos = agent_traj[19]
        distances = []
        for o in padded_others:
            dist = np.linalg.norm(o[-1] - agent_last_pos)
            distances.append(dist)
        distances = np.array(distances)
        closest_idx = np.argsort(distances)[:N]
        closest_others = padded_others[closest_idx]
    else:
        closest_others = np.zeros((0, 50, 2))
    
# Pad if less than N agents
    if len(closest_others) < N:
        pad = np.zeros((N - len(closest_others), 50, 2))
        closest_others = np.concatenate([closest_others, pad], axis=0)
    
    return obs, future, closest_others

# Test it
loader = ArgoverseForecastingLoader('/mnt/d/argoverse_data/train/data')
seq = loader[0]
obs, future, others = preprocess_sequence(seq)
print("obs shape:", obs.shape)
print("future shape:", future.shape)
print("others shape:", others.shape)