import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

class PolarTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        if self.is_train:
            self.train_transforms = torch.nn.Sequential(
                T.RandomHorizontalFlip(p=0.5),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            )

    def __call__(self, img_tensor):
        # img_tensor shape: (1, 64, 360)
        # random shift along theta axis (axis 2)
        if self.is_train and torch.rand(1).item() < 0.5:
            shift = torch.randint(0, 360, (1,)).item()
            img_tensor = torch.roll(img_tensor, shifts=shift, dims=2)
        
        if self.is_train:
            img_tensor = self.train_transforms(img_tensor)

        return img_tensor


class WaferDataset(Dataset):
    def __init__(self, npy_path, label_path, task_id=None, valid_indices=None, is_train=True):
        # Derive shape dynamically
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
        
        self.task_groups = {
            1: ['none', 'Center'],
            2: ['Edge-Ring', 'Edge-Loc'],
            3: ['Scratch', 'Loc', 'Random'],
            4: ['Donut', 'Near-full']
        }

        self.labels = np.array([
            self.label_map.get(lbl, 8)
            for lbl in self.labels_df["label"]
        ])
        
        if valid_indices is not None:
            self.valid_indices = np.array(valid_indices)
        elif task_id is not None:
            allowed_labels_str = self.task_groups[task_id]
            allowed_labels_idx = [self.label_map[l] for l in allowed_labels_str]
            mask = np.isin(self.labels, allowed_labels_idx)
            self.valid_indices = np.where(mask)[0]
        else:
            self.valid_indices = np.arange(len(self.labels))
            
        self.transform = PolarTransform(is_train=is_train)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img = self.images[real_idx]
        img = np.expand_dims(img, axis=0)

        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor / 2.0
        img_tensor = self.transform(img_tensor)

        return (
            img_tensor,
            torch.tensor(self.labels[real_idx], dtype=torch.long)
        )


def get_task_splits(npy_path, label_path, task_id, seed=42, val_frac=0.15, test_frac=0.15):
    """
    3-way stratified split for a single task.
    Returns (abs_train_idx, abs_val_idx, abs_test_idx) as absolute dataset indices.
    test_frac of total is carved out first and never used during training.
    val_frac of total comes from the remaining train+val pool.
    """
    full_ds = WaferDataset(npy_path, label_path, task_id=task_id, is_train=False)
    all_local = np.arange(len(full_ds))
    labels = full_ds.labels[full_ds.valid_indices]
    stratify = labels if len(np.unique(labels)) > 1 else None

    # Step 1: carve out held-out test set
    tv_local, test_local = train_test_split(
        all_local, test_size=test_frac, stratify=stratify, random_state=seed)

    # Step 2: split remaining so val ≈ val_frac of total (0.15/0.85 ≈ 0.176)
    val_adj = val_frac / (1.0 - test_frac)
    tv_labels = labels[tv_local] if stratify is not None else None
    train_local, val_local = train_test_split(
        tv_local, test_size=val_adj, stratify=tv_labels, random_state=seed)

    return (
        full_ds.valid_indices[train_local],
        full_ds.valid_indices[val_local],
        full_ds.valid_indices[test_local],
    )