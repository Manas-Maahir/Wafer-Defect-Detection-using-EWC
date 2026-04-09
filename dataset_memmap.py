import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class WaferDataset(Dataset):
    def __init__(self, npy_path, label_path):
        # Derive shape dynamically so a size mismatch doesn't silently corrupt reads.
        # Each sample is (64, 360) float32 → 64 * 360 * 4 bytes
        sample_bytes  = 64 * 360 * np.dtype("float32").itemsize
        total_bytes   = os.path.getsize(npy_path)
        num_samples   = total_bytes // sample_bytes

        self.images = np.memmap(
            npy_path,
            dtype="float32",
            mode="r",
            shape=(num_samples, 64, 360)
        )
        self.labels_df = pd.read_csv(label_path)

        self.label_map = {
            'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
            'Loc': 4, 'Random': 5, 'Scratch': 6,
            'Near-full': 7, 'none': 8
        }

        self.labels = np.array([
            self.label_map.get(lbl, 8)
            for lbl in self.labels_df["label"]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.expand_dims(img, axis=0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )