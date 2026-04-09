import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_wafer
import os

class WaferDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
        # Mapping for the 9 failure types
        self.label_map = {
            'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 
            'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wafer_map = row['waferMap']
        
        # Preprocess: Cartesian -> Polar
        polar_strip = preprocess_wafer(wafer_map)
        
        # Add channel dimension
        polar_strip = polar_strip[np.newaxis, :, :] # (1, H, W)
        
        # Label handling
        label_str = row['failureType']
        if isinstance(label_str, np.ndarray):
            label_str = label_str[0][0] # LSWMD quirks
            
        label = self.label_map.get(label_str, 8) # default to 'none' if empty
        
        return torch.from_numpy(polar_strip).float(), torch.tensor(label).long()

def get_task_data(pickle_path, task_id):
    """
    Returns data for a specific task.
    Tasks:
    1: ['none', 'Center']
    2: ['Edge-Ring', 'Edge-Loc']
    3: ['Scratch', 'Loc', 'Random']
    4: ['Donut', 'Near-full']
    """
    df = pd.read_pickle(pickle_path)
    
    # Filter only rows with labels
    df = df[df['failureType'].apply(lambda x: len(x) > 0)]
    
    # Process failureType to string format
    def clean_label(x):
        if isinstance(x, np.ndarray) and len(x) > 0 and len(x[0]) > 0:
            return x[0][0]
        return 'none'
    
    df['clean_label'] = df['failureType'].apply(clean_label)
    
    task_groups = {
        1: ['none', 'Center'],
        2: ['Edge-Ring', 'Edge-Loc'],
        3: ['Scratch', 'Loc', 'Random'],
        4: ['Donut', 'Near-full']
    }
    
    selected_labels = task_groups.get(task_id, [])
    task_df = df[df['clean_label'].isin(selected_labels)]
    
    # Split into train/val
    train_df = task_df.sample(frac=0.8, random_state=42)
    val_df = task_df.drop(train_df.index)
    
    return train_df, val_df

if __name__ == "__main__":
    # Test path
    p = 'c:/Users/manas/Desktop/Projects/Wafer/LSWMD.pkl/LSWMD.pkl'
    if os.path.exists(p):
        train_df, val_df = get_task_data(p, 1)
        print(f"Task 1 Train size: {len(train_df)}")
        ds = WaferDataset(train_df.head(10))
        img, lbl = ds[0]
        print(f"Sample img shape: {img.shape}, label: {lbl}")
