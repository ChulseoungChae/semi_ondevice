"""
Training Pipeline for PVD LSTM Models
- 4 model variants x 3 augmentation levels = 12 configurations
- Dual GPU support (4090 x 2)
- Saves models, scalers, training history
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import prepare_datasets, MODEL_CONFIGS
from models import get_model

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Training hyperparameters
EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 30  # early stopping patience


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

    # MAPE (avoid division by zero)
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


def train_model(model_key: str, aug_factor: int, device: torch.device,
                gpu_id: int = 0) -> dict:
    """Train a single model configuration."""
    config_name = f"model_{model_key}_aug{aug_factor}x"
    print(f"\n{'='*60}")
    print(f"Training: {config_name} on GPU {gpu_id}")
    print(f"  Config: {MODEL_CONFIGS[model_key]['desc']}")
    print(f"  Augmentation: {aug_factor}x")
    print(f"{'='*60}")

    # Prepare data
    train_loader, test_loader, scaler_x, scaler_y, info = prepare_datasets(
        DATA_DIR, model_key, aug_factor=aug_factor
    )

    # Create model
    model = get_model(model_key, info['n_features'], info['n_outputs'])
    model = model.to(device)

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
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': info,
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
        'model_key': model_key,
        'aug_factor': aug_factor,
        'best_epoch': best_epoch + 1,
        'total_epochs': epoch + 1,
        'train_time_sec': elapsed,
        'train_samples': info['train_samples'],
        'test_samples': info['test_samples'],
        'n_features': info['n_features'],
        **metrics,
    }

    return result


def run_all_training():
    """Run all 12 training configurations (4 models x 3 aug levels)."""
    # Check GPU
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"Found {n_gpu} GPUs")
        for i in range(n_gpu):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda:0')
    else:
        print("No GPU found, using CPU")
        device = torch.device('cpu')

    model_keys = ['A', 'B', 'C', 'D']
    aug_factors = [1, 20, 50]

    all_results = []
    total_configs = len(model_keys) * len(aug_factors)
    config_idx = 0

    for model_key in model_keys:
        for aug_factor in aug_factors:
            config_idx += 1
            print(f"\n{'#'*60}")
            print(f"# Configuration {config_idx}/{total_configs}")
            print(f"{'#'*60}")

            result = train_model(model_key, aug_factor, device)
            all_results.append(result)

            # Save intermediate results
            with open(os.path.join(RESULT_DIR, 'all_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25s} {'Aug':>4s} {'MAE':>12s} {'RMSE':>12s} {'R2':>10s} {'MAPE%':>10s} {'Time':>8s}")
    print('-' * 80)

    for r in sorted(all_results, key=lambda x: x['RMSE']):
        print(f"{r['config_name']:<25s} {r['aug_factor']:>3d}x "
              f"{r['MAE']:>12.2f} {r['RMSE']:>12.2f} {r['R2']:>10.6f} "
              f"{r['MAPE']:>9.4f}% {r['train_time_sec']:>7.1f}s")

    # Find best
    best = min(all_results, key=lambda x: x['RMSE'])
    print(f"\nBest model: {best['config_name']} (RMSE={best['RMSE']:.2f}, R2={best['R2']:.6f})")

    return all_results


if __name__ == '__main__':
    results = run_all_training()
