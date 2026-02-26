#!/usr/bin/env python3
"""
병렬로 모든 PVD 모델 학습
- GPU 2개 사용: PVD1,2는 GPU0, PVD3,4는 GPU1
"""

import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def train_pvd(pvd_name: str, gpu_id: int, model_type: str = 'lstm'):
    """단일 PVD 모델 학습 (subprocess)"""
    script_path = os.path.join(os.path.dirname(__file__), 'pvd_predictor.py')
    cmd = [
        sys.executable, script_path,
        '--pvd', pvd_name,
        '--model', model_type,
        '--gpu', str(gpu_id)
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"Starting training for {pvd_name} on GPU {gpu_id}")
    result = subprocess.run(cmd, env=env, capture_output=False)
    return pvd_name, result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'patchtst'])
    args = parser.parse_args()

    # GPU 할당
    # PVD1, PVD2 -> GPU 0
    # PVD3, PVD4 -> GPU 1
    tasks = [
        ('PVD1', 0),
        ('PVD2', 0),
        ('PVD3', 1),
        ('PVD4', 1),
    ]

    # 순차적으로 학습 (메모리 관리를 위해)
    # GPU당 2개씩 순차 학습
    for pvd_name, gpu_id in tasks:
        print(f"\n{'='*60}")
        print(f"Training {pvd_name} on GPU {gpu_id}")
        print(f"{'='*60}\n")
        train_pvd(pvd_name, gpu_id, args.model)

    print("\n" + "="*60)
    print("All training completed!")
    print("="*60)


if __name__ == '__main__':
    main()
