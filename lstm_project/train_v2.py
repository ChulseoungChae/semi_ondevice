"""
Training Pipeline V2: 1D CNN vs LSTM Comparison
- 8 input features (no PWPDS.Data in input)
- predict_ahead=0 (predict last timestep of window)
- 6 configurations: CNN x (1x, 20x, 50x) + LSTM x (1x, 20x, 50x)
"""

import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from data_loader import load_all_csvs, create_windows, augment_data, PVDDataset
from models_v2 import PVD1DCNNModel, PVDLSTMModelV2

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# V2 feature configuration
V2_INPUT_COLS = [
    'SBRF5.SetPower', 'EN4.Power', 'Ar.MFC.i',
    'PLA5.Match.Tune.Posi', 'PLA5.Match.DCBias', 'ULVAC.Stage1.Temp1',
    'EN4.Volt', 'Ion.Gauge.i',
]
V2_OUTPUT_COLS = ['PWPDS.Data']
V2_PREDICT_AHEAD = 0
V2_INPUT_WINDOW = 10

# Training hyperparameters
EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 30


def prepare_v2_datasets(aug_factor: int = 1):
    """Prepare train/test data for V2 with 8 features and predict_ahead=0."""
    all_data = load_all_csvs(DATA_DIR)

    train_X_list, train_y_list = [], []
    test_X_list, test_y_list = [], []

    # Group by recipe
    recipes = {}
    for d in all_data:
        key = f"DC{d['dc_setting']}_RF{d['rf_setting']}"
        if key not in recipes:
            recipes[key] = []
        recipes[key].append(d)

    for recipe_key, processes in recipes.items():
        processes.sort(key=lambda x: x['process_num'])

        for i, proc in enumerate(processes):
            X, y = create_windows(
                proc['data'], V2_INPUT_COLS, V2_OUTPUT_COLS,
                input_window=V2_INPUT_WINDOW, predict_ahead=V2_PREDICT_AHEAD,
            )
            if len(X) == 0:
                continue

            if i < len(processes) - 1:
                train_X_list.append(X)
                train_y_list.append(y)
            else:
                test_X_list.append(X)
                test_y_list.append(y)

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    test_X = np.concatenate(test_X_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)

    print(f"V2 data: Train={len(train_X)}, Test={len(test_X)}")

    # Fit scalers
    n_train, seq_len, n_features = train_X.shape
    n_outputs = train_y.shape[1] if train_y.ndim > 1 else 1

    scaler_x = StandardScaler()
    scaler_x.fit(train_X.reshape(-1, n_features))

    scaler_y = StandardScaler()
    scaler_y.fit(train_y.reshape(-1, n_outputs))

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
        'aug_factor': aug_factor,
        'n_features': n_features,
        'n_outputs': n_outputs,
        'train_samples': len(train_X_scaled),
        'test_samples': len(test_X_scaled),
        'input_cols': V2_INPUT_COLS,
        'output_cols': V2_OUTPUT_COLS,
        'predict_ahead': V2_PREDICT_AHEAD,
    }

    return train_loader, test_loader, scaler_x, scaler_y, info


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_batches
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return avg_loss, preds, targets


def compute_metrics(preds_scaled, targets_scaled, scaler_y):
    """Compute metrics in original scale."""
    preds = scaler_y.inverse_transform(preds_scaled)
    targets = scaler_y.inverse_transform(targets_scaled)

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    mask = np.abs(targets) > 1e-6
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    else:
        mape = float('inf')

    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape),
    }


def create_model(model_type: str, n_features: int, n_outputs: int):
    """Create CNN or LSTM model."""
    if model_type == 'CNN':
        return PVD1DCNNModel(n_features=n_features, n_outputs=n_outputs)
    elif model_type == 'LSTM':
        return PVDLSTMModelV2(n_features=n_features, n_outputs=n_outputs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(model_type: str, aug_factor: int, device: torch.device) -> dict:
    """Train a single V2 model configuration."""
    config_name = f"v2_{model_type}_aug{aug_factor}x"
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"  Model: {model_type}, Augmentation: {aug_factor}x")
    print(f"  Features: {len(V2_INPUT_COLS)} ({', '.join(V2_INPUT_COLS)})")
    print(f"  predict_ahead={V2_PREDICT_AHEAD}")
    print(f"{'='*60}")

    # Prepare data
    train_loader, test_loader, scaler_x, scaler_y, info = prepare_v2_datasets(aug_factor)

    # Create model
    model = create_model(model_type, info['n_features'], info['n_outputs'])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': info,
                'model_type': model_type,
            }, os.path.join(MODEL_DIR, f'{config_name}_best.pt'))
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} (ep{best_epoch+1}) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (best at epoch {best_epoch+1})")
            break

    elapsed = time.time() - start_time

    # Load best model and evaluate
    checkpoint = torch.load(os.path.join(MODEL_DIR, f'{config_name}_best.pt'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(test_preds, test_targets, scaler_y)

    print(f"\n  Final Metrics ({config_name}):")
    print(f"    MAE:  {metrics['MAE']:.2f}")
    print(f"    RMSE: {metrics['RMSE']:.2f}")
    print(f"    R2:   {metrics['R2']:.6f}")
    print(f"    MAPE: {metrics['MAPE']:.4f}%")
    print(f"    Training time: {elapsed:.1f}s")

    # Save scalers
    with open(os.path.join(MODEL_DIR, f'{config_name}_scaler_x.pkl'), 'wb') as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(MODEL_DIR, f'{config_name}_scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)

    # Save history
    with open(os.path.join(RESULT_DIR, f'{config_name}_history.json'), 'w') as f:
        json.dump(history, f)

    result = {
        'config_name': config_name,
        'model_type': model_type,
        'aug_factor': aug_factor,
        'best_epoch': best_epoch + 1,
        'total_epochs': epoch + 1,
        'train_time_sec': elapsed,
        'train_samples': info['train_samples'],
        'test_samples': info['test_samples'],
        'n_features': info['n_features'],
        'total_params': total_params,
        **metrics,
    }

    return result


def run_all_v2_training():
    """Run all 6 training configurations: CNN x (1x,20x,50x) + LSTM x (1x,20x,50x)."""
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"Found {n_gpu} GPUs")
        for i in range(n_gpu):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda:0')
    else:
        print("No GPU found, using CPU")
        device = torch.device('cpu')

    model_types = ['CNN', 'LSTM']
    aug_factors = [1, 20, 50]

    all_results = []
    total_configs = len(model_types) * len(aug_factors)
    config_idx = 0

    for model_type in model_types:
        for aug_factor in aug_factors:
            config_idx += 1
            print(f"\n{'#'*60}")
            print(f"# Configuration {config_idx}/{total_configs}: {model_type} x {aug_factor}x aug")
            print(f"{'#'*60}")

            result = train_model(model_type, aug_factor, device)
            all_results.append(result)

            # Save intermediate results
            with open(os.path.join(RESULT_DIR, 'v2_all_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*90}")
    print("V2 TRAINING SUMMARY: 1D CNN vs LSTM (8 features, predict_ahead=0)")
    print(f"{'='*90}")
    print(f"{'Config':<25s} {'Params':>10s} {'MAE':>12s} {'RMSE':>12s} {'R2':>10s} {'MAPE%':>10s} {'Time':>8s}")
    print('-' * 90)

    for r in sorted(all_results, key=lambda x: x['RMSE']):
        print(f"{r['config_name']:<25s} {r['total_params']:>10,d} "
              f"{r['MAE']:>12.2f} {r['RMSE']:>12.2f} {r['R2']:>10.6f} "
              f"{r['MAPE']:>9.4f}% {r['train_time_sec']:>7.1f}s")

    best = min(all_results, key=lambda x: x['RMSE'])
    print(f"\nBest model: {best['config_name']} (RMSE={best['RMSE']:.2f}, R2={best['R2']:.6f})")

    return all_results


if __name__ == '__main__':
    results = run_all_v2_training()
