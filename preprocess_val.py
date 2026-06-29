from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from preprocess_fn import preprocess_sequence
import numpy as np
from tqdm import tqdm

loader = ArgoverseForecastingLoader('/mnt/d/argoverse_data/val/data')

all_obs = []
all_future = []
all_others = []

for i in tqdm(range(len(loader))):
    seq = loader[i]
    obs, future, others = preprocess_sequence(seq)
    all_obs.append(obs)
    all_future.append(future)
    all_others.append(others)

all_obs = np.array(all_obs)
all_future = np.array(all_future)
all_others = np.array(all_others)

print("val_obs shape:", all_obs.shape)
print("val_future shape:", all_future.shape)
print("val_others shape:", all_others.shape)

np.save('/mnt/d/argoverse_data/val_obs.npy', all_obs)
np.save('/mnt/d/argoverse_data/val_future.npy', all_future)
np.save('/mnt/d/argoverse_data/val_others.npy', all_others)

print("Saved!")