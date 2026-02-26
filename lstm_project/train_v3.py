"""
Training Pipeline V3: Full Timeline (no active filtering)
- Raw CSV data as-is, no active row filtering
- All CSVs concatenated → windows → augment → train
- 6 configurations: CNN x (1x, 20x, 50x) + LSTM x (1x, 20x, 50x)
"""

import os
import glob
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from data_loader import parse_recipe_name, augment_data, PVDDataset
from models_v2 import PVD1DCNNModel, PVDLSTMModelV2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

V3_INPUT_COLS = [
    'SBRF5.SetPower', 'EN4.Power', 'Ar.MFC.i',
    'PLA5.Match.Tune.Posi', 'PLA5.Match.DCBias', 'ULVAC.Stage1.Temp1',
    'EN4.Volt', 'Ion.Gauge.i',
]
V3_OUTPUT_COLS = ['PWPDS.Data']
V3_INPUT_WINDOW = 10
V3_PREDICT_AHEAD = 0

EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 30


def load_all_csvs_full(data_dir: str):
    """Load all CSV files WITHOUT active filtering."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    all_data = []
    for f in files:
        recipe = parse_recipe_name(f)
        df = pd.read_csv(f)
        recipe['data'] = df
        recipe['total_rows'] = len(df)
        all_data.append(recipe)
    print(f"Loaded {len(all_data)} files (full timeline, no filtering)")
    return all_data


def create_windows(data: pd.DataFrame, input_cols, output_cols,
                   input_window=V3_INPUT_WINDOW, predict_ahead=V3_PREDICT_AHEAD):
    """Create sliding windows from full data."""
    input_data = data[input_cols].values.astype(np.float64)
    output_data = data[output_cols].values.astype(np.float64)
    n = len(data)

    X_list, y_list = [], []
    for i in range(n - input_window - predict_ahead + 1):
        X_list.append(input_data[i:i + input_window])
        y_list.append(output_data[i + input_window + predict_ahead - 1])

    if len(X_list) == 0:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)


def prepare_v3_datasets(aug_factor: int = 1):
    """Prepare train/test data: full timeline, recipe-based split."""
    all_data = load_all_csvs_full(DATA_DIR)

    # Group by recipe
    recipes = {}
    for d in all_data:
        key = f"DC{d['dc_setting']}_RF{d['rf_setting']}"
        if key not in recipes:
            recipes[key] = []
        recipes[key].append(d)

    train_X_list, train_y_list = [], []
    test_X_list, test_y_list = [], []

    for recipe_key, processes in recipes.items():
        processes.sort(key=lambda x: x['process_num'])
        for i, proc in enumerate(processes):
            X, y = create_windows(proc['data'], V3_INPUT_COLS, V3_OUTPUT_COLS)
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

    print(f"V3 data: Train={len(train_X)}, Test={len(test_X)}")

    n_train, seq_len, n_features = train_X.shape
    n_outputs = train_y.shape[1] if train_y.ndim > 1 else 1

    scaler_x = StandardScaler()
    scaler_x.fit(train_X.reshape(-1, n_features))
    scaler_y = StandardScaler()
    scaler_y.fit(train_y.reshape(-1, n_outputs))

    train_X_sc = scaler_x.transform(train_X.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    test_X_sc = scaler_x.transform(test_X.reshape(-1, n_features)).reshape(len(test_X), seq_len, n_features)
    train_y_sc = scaler_y.transform(train_y.reshape(-1, n_outputs)).reshape(-1, n_outputs)
    test_y_sc = scaler_y.transform(test_y.reshape(-1, n_outputs)).reshape(-1, n_outputs)

    if aug_factor > 1:
        train_X_sc, train_y_sc = augment_data(train_X_sc, train_y_sc, aug_factor)
        print(f"  After {aug_factor}x augmentation: {len(train_X_sc)} train samples")

    train_loader = DataLoader(PVDDataset(train_X_sc, train_y_sc),
                              batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(PVDDataset(test_X_sc, test_y_sc),
                             batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    info = {
        'aug_factor': aug_factor,
        'n_features': n_features,
        'n_outputs': n_outputs,
        'train_samples': len(train_X_sc),
        'test_samples': len(test_X_sc),
        'input_cols': V3_INPUT_COLS,
        'output_cols': V3_OUTPUT_COLS,
        'predict_ahead': V3_PREDICT_AHEAD,
        'full_timeline': True,
    }
    return train_loader, test_loader, scaler_x, scaler_y, info


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
        pred = model(X_b)
        total_loss += criterion(pred, y_b).item()
        n += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_b.cpu().numpy())
    return total_loss / n, np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(preds_sc, targets_sc, scaler_y):
    preds = scaler_y.inverse_transform(preds_sc)
    targets = scaler_y.inverse_transform(targets_sc)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    mask = np.abs(targets) > 1e-6
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100 if mask.sum() > 0 else float('inf')
    return {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2), 'MAPE': float(mape)}


def train_model(model_type: str, aug_factor: int, device: torch.device) -> dict:
    config_name = f"v3_{model_type}_aug{aug_factor}x"
    print(f"\n{'='*60}")
    print(f"Training: {config_name} (full timeline)")
    print(f"{'='*60}")

    train_loader, test_loader, scaler_x, scaler_y, info = prepare_v3_datasets(aug_factor)

    if model_type == 'CNN':
        model = PVD1DCNNModel(n_features=info['n_features'], n_outputs=info['n_outputs'])
    else:
        model = PVDLSTMModelV2(n_features=info['n_features'], n_outputs=info['n_outputs'])
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_type}, Params: {total_params:,}, Aug: {aug_factor}x")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, test_loader, criterion, device)
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
                'epoch': epoch, 'val_loss': val_loss,
                'config': info, 'model_type': model_type,
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

    # Load best & final eval
    ckpt = torch.load(os.path.join(MODEL_DIR, f'{config_name}_best.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    _, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(test_preds, test_targets, scaler_y)

    print(f"\n  Final ({config_name}):")
    print(f"    MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
          f"R2={metrics['R2']:.6f}  MAPE={metrics['MAPE']:.4f}%  Time={elapsed:.1f}s")

    # Save artifacts
    with open(os.path.join(MODEL_DIR, f'{config_name}_scaler_x.pkl'), 'wb') as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(MODEL_DIR, f'{config_name}_scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(RESULT_DIR, f'{config_name}_history.json'), 'w') as f:
        json.dump(history, f)

    return {
        'config_name': config_name, 'model_type': model_type,
        'aug_factor': aug_factor, 'best_epoch': best_epoch + 1,
        'total_epochs': epoch + 1, 'train_time_sec': elapsed,
        'train_samples': info['train_samples'], 'test_samples': info['test_samples'],
        'n_features': info['n_features'], 'total_params': total_params,
        **metrics,
    }


def run_all():
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"Found {n_gpu} GPUs")
        for i in range(n_gpu):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda:0')
    else:
        print("No GPU, using CPU")
        device = torch.device('cpu')

    all_results = []
    configs = [('CNN', 1), ('CNN', 20), ('CNN', 50),
               ('LSTM', 1), ('LSTM', 20), ('LSTM', 50)]

    for idx, (mt, af) in enumerate(configs, 1):
        print(f"\n{'#'*60}")
        print(f"# Config {idx}/{len(configs)}: {mt} x {af}x aug (full timeline)")
        print(f"{'#'*60}")
        result = train_model(mt, af, device)
        all_results.append(result)

        with open(os.path.join(RESULT_DIR, 'v3_all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n\n{'='*95}")
    print("V3 TRAINING SUMMARY: Full Timeline (no active filtering)")
    print(f"{'='*95}")
    print(f"{'Config':<25s} {'Params':>10s} {'MAE':>12s} {'RMSE':>12s} {'R2':>10s} {'MAPE%':>10s} {'Time':>8s}")
    print('-' * 95)
    for r in sorted(all_results, key=lambda x: x['RMSE']):
        print(f"{r['config_name']:<25s} {r['total_params']:>10,d} "
              f"{r['MAE']:>12.2f} {r['RMSE']:>12.2f} {r['R2']:>10.6f} "
              f"{r['MAPE']:>9.4f}% {r['train_time_sec']:>7.1f}s")

    best = min(all_results, key=lambda x: x['RMSE'])
    print(f"\nBest: {best['config_name']} (RMSE={best['RMSE']:.2f}, R2={best['R2']:.6f})")
    return all_results


if __name__ == '__main__':
    run_all()
