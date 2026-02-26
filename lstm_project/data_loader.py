"""
PVD4 PDS Data Loader & Preprocessor
- 45 CSV files from PDS_Data_Log
- Sliding window: 10s input -> 5s ahead prediction
- Data augmentation: 1x, 20x, 50x
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


# 4 model input configurations
MODEL_CONFIGS = {
    'A': {
        'input_cols': ['Ar.MFC.i', 'EN4.Power', 'SBRF5.SetPower', 'PLA5.Match.DCBias'],
        'output_cols': ['PWPDS.Data'],
        'desc': '4 features (no PWPDS.Data input)'
    },
    'B': {
        'input_cols': ['Ar.MFC.i', 'EN4.Power', 'SBRF5.SetPower', 'PLA5.Match.DCBias', 'PWPDS.Data'],
        'output_cols': ['PWPDS.Data'],
        'desc': '5 features (with PWPDS.Data input)'
    },
    'C': {
        'input_cols': ['Ar.MFC.i', 'EN4.Power', 'EN4.Current', 'EN4.Volt',
                       'PLA5.Match.Load.Posi', 'PLA5.Match.Tune.Posi',
                       'PLA5.Match.Load.Pre', 'PLA5.Match.Tune.Pre',
                       'PLA5.Match.DCBias', 'SBRF5.Forward', 'SBRF5.Reflect',
                       'SBRF5.SetPower'],
        'output_cols': ['PWPDS.Data'],
        'desc': '12 features (no PWPDS.Data input)'
    },
    'D': {
        'input_cols': ['Ar.MFC.i', 'EN4.Power', 'EN4.Current', 'EN4.Volt',
                       'PLA5.Match.Load.Posi', 'PLA5.Match.Tune.Posi',
                       'PLA5.Match.Load.Pre', 'PLA5.Match.Tune.Pre',
                       'PLA5.Match.DCBias', 'SBRF5.Forward', 'SBRF5.Reflect',
                       'SBRF5.SetPower', 'PWPDS.Data'],
        'output_cols': ['PWPDS.Data'],
        'desc': '13 features (with PWPDS.Data input)'
    },
}

INPUT_WINDOW = 10   # 10 seconds input window
PREDICT_AHEAD = 5   # predict 5 seconds ahead


def parse_recipe_name(filename: str) -> dict:
    """Parse recipe info from filename like DC1000RF300(1).csv"""
    name = os.path.basename(filename).replace('.csv', '')

    info = {'filename': name, 'dc_setting': 0, 'rf_setting': 0, 'process_num': 1,
            'recipe_type': 'unknown'}

    # DCignRFxxx(n)
    m = re.match(r'DCignRF(\d+)\((\d+)\)', name)
    if m:
        info['rf_setting'] = int(m.group(1))
        info['process_num'] = int(m.group(2))
        info['recipe_type'] = 'RF_only'
        return info

    # DCxxxxRFxxx(n)
    m = re.match(r'DC(\d+)RF(\d+)\((\d+)\)', name)
    if m:
        info['dc_setting'] = int(m.group(1))
        info['rf_setting'] = int(m.group(2))
        info['process_num'] = int(m.group(3))
        info['recipe_type'] = 'DC_RF'
        return info

    # DC3000R300(2) - typo in filename
    m = re.match(r'DC(\d+)R(\d+)\((\d+)\)', name)
    if m:
        info['dc_setting'] = int(m.group(1))
        info['rf_setting'] = int(m.group(2))
        info['process_num'] = int(m.group(3))
        info['recipe_type'] = 'DC_RF'
        return info

    # DCxxxx(n)
    m = re.match(r'DC(\d+)\((\d+)\)', name)
    if m:
        info['dc_setting'] = int(m.group(1))
        info['process_num'] = int(m.group(2))
        info['recipe_type'] = 'DC_only'
        return info

    return info


def load_all_csvs(data_dir: str) -> List[dict]:
    """Load all CSV files with recipe metadata."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    all_data = []

    for f in files:
        recipe = parse_recipe_name(f)
        df = pd.read_csv(f)

        # Filter to active process rows (DC or RF is on)
        active_mask = (df['EN4.Power'] > 0) | (df['SBRF5.SetPower'] > 0)
        df_active = df[active_mask].reset_index(drop=True)

        if len(df_active) < INPUT_WINDOW + PREDICT_AHEAD:
            print(f"  Warning: {recipe['filename']} has only {len(df_active)} active rows, skipping")
            continue

        recipe['data'] = df_active
        recipe['total_rows'] = len(df_active)
        all_data.append(recipe)

    print(f"Loaded {len(all_data)} process files")
    return all_data


def create_windows(data: pd.DataFrame, input_cols: List[str], output_cols: List[str],
                   input_window: int = INPUT_WINDOW, predict_ahead: int = PREDICT_AHEAD
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window samples.
    Input: [t-9, t-8, ..., t] (10 timesteps)
    Output: value at t+5
    """
    X_list, y_list = [], []
    input_data = data[input_cols].values.astype(np.float64)
    output_data = data[output_cols].values.astype(np.float64)

    n = len(data)
    for i in range(n - input_window - predict_ahead + 1):
        X_list.append(input_data[i:i + input_window])
        y_list.append(output_data[i + input_window + predict_ahead - 1])

    if len(X_list) == 0:
        return np.array([]), np.array([])

    return np.array(X_list), np.array(y_list)


def augment_data(X: np.ndarray, y: np.ndarray, factor: int,
                 noise_std: float = 0.02, time_warp_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Augment time series data.
    Methods:
    1. Jittering: add Gaussian noise (scaled per feature)
    2. Scaling: random scaling per sample
    3. Magnitude warping: smooth magnitude changes
    """
    if factor <= 1:
        return X, y

    aug_X = [X]
    aug_y = [y]
    rng = np.random.RandomState(42)

    for k in range(factor - 1):
        # Mix of augmentation methods
        method = k % 3

        if method == 0:
            # Jittering: add small Gaussian noise
            feature_std = np.std(X, axis=(0, 1), keepdims=True)
            feature_std = np.where(feature_std == 0, 1.0, feature_std)
            noise_x = rng.normal(0, noise_std, X.shape) * feature_std
            noise_y_val = rng.normal(0, noise_std, y.shape) * np.std(y, axis=0, keepdims=True)
            aug_X.append(X + noise_x)
            aug_y.append(y + noise_y_val)

        elif method == 1:
            # Scaling: random scale factor per sample
            scale = rng.normal(1.0, 0.05, size=(X.shape[0], 1, 1))
            aug_X.append(X * scale)
            scale_y = scale[:, 0, :y.shape[1]] if y.ndim > 1 else scale[:, 0, 0]
            aug_y.append(y * scale_y)

        else:
            # Magnitude warping with smooth curve
            seq_len = X.shape[1]
            # Generate smooth random curve
            knots = rng.normal(1.0, 0.1, size=(X.shape[0], 4))
            warp = np.ones((X.shape[0], seq_len))
            for j in range(4):
                center = seq_len * (j + 0.5) / 4
                sigma = seq_len / 4
                for t in range(seq_len):
                    warp[:, t] *= 1 + (knots[:, j] - 1) * np.exp(-0.5 * ((t - center) / sigma) ** 2)
            warp_x = warp[:, :, np.newaxis]
            aug_X.append(X * warp_x)
            # Apply average warp to y
            avg_warp = np.mean(warp, axis=1, keepdims=True)
            if y.ndim == 1:
                aug_y.append(y * avg_warp[:, 0])
            else:
                aug_y.append(y * avg_warp)

    X_aug = np.concatenate(aug_X, axis=0)
    y_aug = np.concatenate(aug_y, axis=0)

    # Shuffle
    idx = rng.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


class PVDDataset(Dataset):
    """PyTorch Dataset for PVD time series."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_datasets(data_dir: str, model_key: str, aug_factor: int = 1,
                     train_ratio: float = 0.8, seed: int = 42
                     ) -> Tuple[DataLoader, DataLoader, StandardScaler, StandardScaler, dict]:
    """Prepare train/test DataLoaders for a specific model configuration.

    Returns: train_loader, test_loader, scaler_x, scaler_y, info_dict
    """
    config = MODEL_CONFIGS[model_key]
    input_cols = config['input_cols']
    output_cols = config['output_cols']

    all_data = load_all_csvs(data_dir)

    # Split by recipe: use 2 of 3 processes for train, 1 for test per recipe
    train_X_list, train_y_list = [], []
    test_X_list, test_y_list = [], []

    # Group by recipe type
    recipes = {}
    for d in all_data:
        key = f"DC{d['dc_setting']}_RF{d['rf_setting']}"
        if key not in recipes:
            recipes[key] = []
        recipes[key].append(d)

    for recipe_key, processes in recipes.items():
        # Sort by process number
        processes.sort(key=lambda x: x['process_num'])

        for i, proc in enumerate(processes):
            X, y = create_windows(proc['data'], input_cols, output_cols)
            if len(X) == 0:
                continue

            if i < len(processes) - 1:  # train: first N-1 processes
                train_X_list.append(X)
                train_y_list.append(y)
            else:  # test: last process
                test_X_list.append(X)
                test_y_list.append(y)

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    test_X = np.concatenate(test_X_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)

    print(f"Model {model_key}: Train samples={len(train_X)}, Test samples={len(test_X)}")

    # Fit scalers on training data
    n_train, seq_len, n_features = train_X.shape
    n_outputs = train_y.shape[1] if train_y.ndim > 1 else 1

    scaler_x = StandardScaler()
    train_X_flat = train_X.reshape(-1, n_features)
    scaler_x.fit(train_X_flat)

    scaler_y = StandardScaler()
    train_y_flat = train_y.reshape(-1, n_outputs)
    scaler_y.fit(train_y_flat)

    # Transform
    train_X_scaled = scaler_x.transform(train_X.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    test_X_scaled = scaler_x.transform(test_X.reshape(-1, n_features)).reshape(len(test_X), seq_len, n_features)
    train_y_scaled = scaler_y.transform(train_y.reshape(-1, n_outputs)).reshape(-1, n_outputs)
    test_y_scaled = scaler_y.transform(test_y.reshape(-1, n_outputs)).reshape(-1, n_outputs)

    # Augment training data
    if aug_factor > 1:
        train_X_scaled, train_y_scaled = augment_data(train_X_scaled, train_y_scaled, aug_factor)
        print(f"  After {aug_factor}x augmentation: {len(train_X_scaled)} train samples")

    # Create DataLoaders
    train_dataset = PVDDataset(train_X_scaled, train_y_scaled)
    test_dataset = PVDDataset(test_X_scaled, test_y_scaled)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    info = {
        'model_key': model_key,
        'aug_factor': aug_factor,
        'n_features': n_features,
        'n_outputs': n_outputs,
        'train_samples': len(train_X_scaled),
        'test_samples': len(test_X_scaled),
        'input_cols': input_cols,
        'output_cols': output_cols,
    }

    return train_loader, test_loader, scaler_x, scaler_y, info


def load_all_active_data(data_dir: str) -> pd.DataFrame:
    """Load all active-process data with recipe info for limit analysis."""
    all_data = load_all_csvs(data_dir)
    frames = []
    for d in all_data:
        df = d['data'].copy()
        df['recipe'] = d['filename']
        df['dc_setting'] = d['dc_setting']
        df['rf_setting'] = d['rf_setting']
        df['recipe_type'] = d['recipe_type']
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    data_dir = '/home/goo4168/baco/PDS_Data_Log'

    print("=== Testing data loader ===")
    all_data = load_all_csvs(data_dir)

    print(f"\nTotal files loaded: {len(all_data)}")
    for d in all_data[:3]:
        print(f"  {d['filename']}: type={d['recipe_type']}, DC={d['dc_setting']}, RF={d['rf_setting']}, rows={d['total_rows']}")

    print("\n=== Testing window creation ===")
    for model_key in ['A', 'B', 'C', 'D']:
        config = MODEL_CONFIGS[model_key]
        X, y = create_windows(all_data[0]['data'], config['input_cols'], config['output_cols'])
        print(f"Model {model_key}: X shape={X.shape}, y shape={y.shape}")

    print("\n=== Testing full pipeline (Model A, 1x aug) ===")
    train_loader, test_loader, sx, sy, info = prepare_datasets(data_dir, 'A', aug_factor=1)
    print(f"Info: {info}")
    for X_batch, y_batch in train_loader:
        print(f"Batch X: {X_batch.shape}, y: {y_batch.shape}")
        break
