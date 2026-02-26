"""
v3_CNN LR 튜닝: LR=3e-4로 aug20x, aug50x 학습
기존 대비 변경: LEARNING_RATE 1e-3 → 3e-4
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models_v2 import PVD1DCNNModel
from train_v3 import prepare_v3_datasets, train_one_epoch, evaluate, compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

EPOCHS = 300
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 30


def train_config(aug_factor: int, device: torch.device) -> dict:
    config_name = f"v3_CNN_lr3e4_aug{aug_factor}x"
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"  LR={LEARNING_RATE}, aug={aug_factor}x")
    print(f"{'='*60}")

    train_loader, test_loader, scaler_x, scaler_y, info = prepare_v3_datasets(aug_factor)

    model = PVD1DCNNModel(n_features=info['n_features'], n_outputs=info['n_outputs'])
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

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
                'config': info, 'model_type': 'CNN',
            }, os.path.join(MODEL_DIR, f'{config_name}_best.pt'))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} (ep{best_epoch+1}) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (best at epoch {best_epoch+1})")
            break

    elapsed = time.time() - start_time

    # Load best & eval
    ckpt = torch.load(os.path.join(MODEL_DIR, f'{config_name}_best.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    _, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(test_preds, test_targets, scaler_y)

    print(f"\n  Result ({config_name}):")
    print(f"    Best epoch: {best_epoch+1}/{epoch+1}")
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
        'config_name': config_name, 'aug_factor': aug_factor,
        'lr': LEARNING_RATE, 'best_epoch': best_epoch + 1,
        'total_epochs': epoch + 1, 'train_time_sec': elapsed,
        'train_samples': info['train_samples'], 'test_samples': info['test_samples'],
        **metrics,
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = []
    for aug in [20, 50]:
        r = train_config(aug, device)
        results.append(r)

    # Compare with original
    print(f"\n\n{'='*80}")
    print("비교 결과: LR=3e-4 vs 기존 LR=1e-3")
    print(f"{'='*80}")
    print(f"{'Config':<30s} {'BestEp':>6s} {'MAE':>10s} {'RMSE':>10s} {'R2':>10s} {'MAPE%':>8s}")
    print('-' * 80)

    # Original results
    originals = [
        {'config_name': 'v3_CNN_aug20x (LR=1e-3)', 'best_epoch': 3,
         'MAE': 5724.88, 'RMSE': 11403.87, 'R2': 0.9399, 'MAPE': 0.0027},
        {'config_name': 'v3_CNN_aug50x (LR=1e-3)', 'best_epoch': 5,
         'MAE': 6225.00, 'RMSE': 12051.88, 'R2': 0.9329, 'MAPE': 0.0029},
    ]

    for r in originals + results:
        print(f"{r['config_name']:<30s} {r['best_epoch']:>6d} "
              f"{r['MAE']:>10.2f} {r['RMSE']:>10.2f} {r['R2']:>10.6f} {r['MAPE']:>7.4f}%")

    with open(os.path.join(RESULT_DIR, 'v3_lr_tuning_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n결과 저장: {os.path.join(RESULT_DIR, 'v3_lr_tuning_results.json')}")


if __name__ == '__main__':
    main()
