"""
v3_CNN 전체 데이터 학습: 45개 공정 모두 Train, 레시피별 마지막 공정만 Test 평가
- Train: 전체 45개 공정
- Test: 레시피별 마지막(3번째) 공정 (15개)
- 기존 LR=1e-3, aug20x
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
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import PVDDataset, augment_data
from models_v2 import PVD1DCNNModel
from train_v3 import (load_all_csvs_full, create_windows,
                       V3_INPUT_COLS, V3_OUTPUT_COLS, V3_INPUT_WINDOW, V3_PREDICT_AHEAD,
                       train_one_epoch, evaluate, compute_metrics)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 30


def prepare_alldata(aug_factor: int = 1):
    """전체 45개 공정 → Train, 레시피별 마지막 공정 → Test 평가용."""
    all_data = load_all_csvs_full(DATA_DIR)

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
            # 전체 다 Train에 포함
            train_X_list.append(X)
            train_y_list.append(y)
            # 마지막 공정은 Test 평가용에도 포함
            if i == len(processes) - 1:
                test_X_list.append(X)
                test_y_list.append(y)

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    test_X = np.concatenate(test_X_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)

    print(f"Train: {len(train_X)} windows (전체 45 공정)")
    print(f"Test:  {len(test_X)} windows (레시피별 마지막 공정 15개)")

    n_train, seq_len, n_features = train_X.shape
    n_outputs = train_y.shape[1] if train_y.ndim > 1 else 1

    scaler_x = StandardScaler()
    scaler_x.fit(train_X.reshape(-1, n_features))
    scaler_y = StandardScaler()
    scaler_y.fit(train_y.reshape(-1, n_outputs))

    train_X_sc = scaler_x.transform(train_X.reshape(-1, n_features)).reshape(len(train_X), seq_len, n_features)
    test_X_sc = scaler_x.transform(test_X.reshape(-1, n_features)).reshape(len(test_X), seq_len, n_features)
    train_y_sc = scaler_y.transform(train_y.reshape(-1, n_outputs)).reshape(-1, n_outputs)
    test_y_sc = scaler_y.transform(test_y.reshape(-1, n_outputs)).reshape(-1, n_outputs)

    if aug_factor > 1:
        train_X_sc, train_y_sc = augment_data(train_X_sc, train_y_sc, aug_factor)
        print(f"After {aug_factor}x augmentation: {len(train_X_sc)} train samples")

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
        'all_data_train': True,
    }
    return train_loader, test_loader, scaler_x, scaler_y, info


def train_alldata(aug_factor: int, device: torch.device) -> dict:
    config_name = f"v3_CNN_alldata_aug{aug_factor}x"
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")

    train_loader, test_loader, scaler_x, scaler_y, info = prepare_alldata(aug_factor)

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

    # Save
    with open(os.path.join(MODEL_DIR, f'{config_name}_scaler_x.pkl'), 'wb') as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(MODEL_DIR, f'{config_name}_scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(RESULT_DIR, f'{config_name}_history.json'), 'w') as f:
        json.dump(history, f)

    return {
        'config_name': config_name, 'aug_factor': aug_factor,
        'best_epoch': best_epoch + 1, 'total_epochs': epoch + 1,
        'train_time_sec': elapsed,
        'train_samples': info['train_samples'], 'test_samples': info['test_samples'],
        **metrics,
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = []
    for aug in [20, 50]:
        r = train_alldata(aug, device)
        results.append(r)

    # Compare
    print(f"\n\n{'='*80}")
    print("비교: 전체학습(alldata) vs 기존(2:1 split)")
    print(f"{'='*80}")
    print(f"{'Config':<35s} {'Train':>6s} {'Test':>6s} {'BestEp':>6s} "
          f"{'MAE':>10s} {'RMSE':>10s} {'R2':>10s} {'MAPE%':>8s}")
    print('-' * 95)

    originals = [
        {'config_name': 'v3_CNN_aug20x (기존 2:1)', 'train_samples': 78500,
         'test_samples': 1968, 'best_epoch': 3,
         'MAE': 5724.88, 'RMSE': 11403.87, 'R2': 0.9399, 'MAPE': 0.0027},
        {'config_name': 'v3_CNN_aug50x (기존 2:1)', 'train_samples': 196250,
         'test_samples': 1968, 'best_epoch': 5,
         'MAE': 6225.00, 'RMSE': 12051.88, 'R2': 0.9329, 'MAPE': 0.0029},
    ]

    for r in originals + results:
        print(f"{r['config_name']:<35s} {r['train_samples']:>6d} {r['test_samples']:>6d} "
              f"{r['best_epoch']:>6d} {r['MAE']:>10.2f} {r['RMSE']:>10.2f} "
              f"{r['R2']:>10.6f} {r['MAPE']:>7.4f}%")

    with open(os.path.join(RESULT_DIR, 'v3_alldata_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
