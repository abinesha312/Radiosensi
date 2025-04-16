import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

def generate_large_synthetic_data(data_dir='data/processed', n_samples=1_000_000):
    print(f"Generating large synthetic dataset with {n_samples} samples...")
    np.random.seed(42)
    baseline = np.random.normal(5.2, 0.7, n_samples)
    is_sensitive = np.random.binomial(1, 0.4, n_samples)
    h24 = baseline * (1 - np.where(is_sensitive, 0.12, 0.05) * np.random.normal(1, 0.3, n_samples))
    h72 = h24 * (1 - np.where(is_sensitive, 0.18, 0.07) * np.random.normal(1, 0.2, n_samples))
    d10 = h72 * (1 + np.where(is_sensitive, 0.05, 0.12) * np.random.normal(1, 0.2, n_samples))
    noise_factor = 0.15
    baseline += np.random.normal(0, noise_factor * baseline.std(), n_samples)
    h24 += np.random.normal(0, noise_factor * h24.std(), n_samples)
    h72 += np.random.normal(0, noise_factor * h72.std(), n_samples)
    d10 += np.random.normal(0, noise_factor * d10.std(), n_samples)
    age = np.random.normal(60, 10, n_samples)
    df = pd.DataFrame({
        'patient_id': [f'P{i:07d}' for i in range(n_samples)],
        'baseline': baseline,
        '24h': h24,
        '72h': h72,
        '10d': d10,
        'age': age,
        'radiosensitive': is_sensitive
    })
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, 'telomere_data_large.csv'), index=False)
    print(f"Saved large synthetic dataset to {data_dir}/telomere_data_large.csv")

class TelomereDataset(Dataset):
    def __init__(self, data_file='data/processed/telomere_data_large.csv', transform=None):
        self.transform = transform
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Extract telomere measurements
        telomere_features = self.data.iloc[idx][['baseline', '24h', '72h', '10d']].values.astype(np.float32)
        # Calculate derived features
        deltas = np.diff(telomere_features)  # Rate of change (3,)
        acceleration = np.diff(np.append(deltas, 0))  # Acceleration (2,)
        variability = np.std(telomere_features)  # Variability (1,)
        # Combine all features (4+3+2+1+1=11)
        features = np.concatenate([
            telomere_features,  # (4,)
            deltas,             # (3,)
            [acceleration[0], acceleration[1]],  # (2,)
            [variability],      # (1,)
            [self.data.iloc[idx]['age']]  # (1,)
        ]).astype(np.float32)
        label = np.array([self.data.iloc[idx]['radiosensitive']], dtype=np.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return {'features': features_tensor, 'labels': label_tensor}

if __name__ == "__main__":
    generate_large_synthetic_data()
