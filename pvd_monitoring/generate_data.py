#!/usr/bin/env python3
"""
PVD 공정 실시간 데이터 생성기

PDS_Data_Log의 실제 CSV 파일을 기반으로 1공정씩 시뮬레이션.
- 정상 공정: 원본 CSV 그대로 1초 간격으로 재생
- 비정상 공정: PWPDS.Data를 의도적으로 상/하한 벗어나게 주입
"""

import os
import sys
import time
import random
import glob
import json
import argparse
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 원본 데이터 경로
PDS_DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
BOUNDS_PATH = '/home/goo4168/baco/lstm_project/results/bounds_lookup.json'

# 출력 칼럼 (PDS_Data_Log CSV와 동일)
OUTPUT_COLUMNS = [
    'Timer', 'ULVAC.Stage1.Temp1', 'ULVAC.Stage2.Temp1', 'EN4.Power', 'EN4.Current',
    'EN4.Volt', 'PLA5.Match.Load.Posi', 'PLA5.Match.Tune.Posi', 'PLA5.Match.Load.Pre',
    'PLA5.Match.Tune.Pre', 'PLA5.Match.DCBias', 'SBRF5.Forward', 'SBRF5.Reflect',
    'SBRF5.SetPower', 'PWESC.Volt1', 'PWESC.Volt2', 'OES.Data6', 'Line.Gauge.i',
    'Ion.Gauge.i', 'Baratron.Gauge.i', 'Ar.MFC.i', 'Ar2.MFC.i', 'Ar.MFC.o', 'Ar2.MFC.o',
    'PWPDS.Data', 'PWPDS.Data2', 'PWPDS.Data3', 'PWPDS.Data4',
    'PWPDS.Data5', 'PWPDS.Data6', 'PWPDS.Data7', 'PWPDS.Data8'
]


def parse_recipe(filename: str) -> dict:
    """파일명에서 레시피 정보 추출"""
    name = os.path.basename(filename).replace('.csv', '')
    info = {'dc_setting': 0, 'rf_setting': 0}

    m = re.match(r'DCignRF(\d+)\(\d+\)', name)
    if m:
        info['rf_setting'] = int(m.group(1))
        return info

    m = re.match(r'DC(\d+)RF(\d+)\(\d+\)', name)
    if m:
        info['dc_setting'] = int(m.group(1))
        info['rf_setting'] = int(m.group(2))
        return info

    m = re.match(r'DC(\d+)R(\d+)\(\d+\)', name)
    if m:
        info['dc_setting'] = int(m.group(1))
        info['rf_setting'] = int(m.group(2))
        return info

    m = re.match(r'DC(\d+)\(\d+\)', name)
    if m:
        info['dc_setting'] = int(m.group(1))
        return info

    return info


def load_bounds() -> dict:
    """상/하한 경계값 로드"""
    if os.path.exists(BOUNDS_PATH):
        with open(BOUNDS_PATH, 'r') as f:
            return json.load(f)
    return {}


def get_recipe_key(dc_setting: int, rf_setting: int) -> str:
    return f"DC{dc_setting}_RF{rf_setting}"


class PVDDataGenerator:
    """PDS_Data_Log 기반 데이터 생성기"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 원본 CSV 로드
        self.csv_files = sorted(glob.glob(os.path.join(PDS_DATA_DIR, '*.csv')))
        print(f"[INFO] 원본 CSV 파일 {len(self.csv_files)}개 발견")

        # 상하한 로드
        self.bounds = load_bounds()
        print(f"[INFO] 상하한 경계값 {len(self.bounds)}개 레시피 로드")

    def generate_process(self, mode: str = 'normal') -> Tuple[str, str]:
        """
        1공정 데이터 생성.
        mode: 'normal' 또는 'abnormal'
        """
        # 랜덤 CSV 선택
        csv_file = random.choice(self.csv_files)
        recipe = parse_recipe(csv_file)
        recipe_key = get_recipe_key(recipe['dc_setting'], recipe['rf_setting'])

        df = pd.read_csv(csv_file)
        filename_base = os.path.basename(csv_file).replace('.csv', '')

        print(f"[INFO] 선택된 파일: {filename_base}")
        print(f"[INFO] 레시피: DC={recipe['dc_setting']}, RF={recipe['rf_setting']} ({recipe_key})")
        print(f"[INFO] 모드: {mode}")

        # 공정 ID 및 출력 파일
        now = datetime.now()
        process_id = now.strftime('%Y%m%d_%H%M%S')
        out_filename = f"PVD4_{now.strftime('%y%m%d')}_{process_id}.csv"
        out_path = os.path.join(self.output_dir, out_filename)

        # 비정상 공정: 이상 주입 구간 설정
        anomaly_indices = set()
        ar_fluctuation_indices = set()
        if mode == 'abnormal':
            anomaly_indices = self._plan_anomalies(df, recipe_key)
            ar_fluctuation_indices = self._plan_ar_fluctuation(df)
            print(f"[INFO] PWPDS 이상 주입 인덱스: {sorted(anomaly_indices)}")
            print(f"[INFO] Ar 진동 주입 인덱스: {sorted(ar_fluctuation_indices)}")

        # 데이터 생성 (1초에 1행씩)
        with open(out_path, 'w') as f:
            f.write(','.join(OUTPUT_COLUMNS) + '\n')
            print(f"[INFO] 공정 시작: {process_id} -> {out_path}")

            for idx in range(len(df)):
                current_time = datetime.now()
                row = df.iloc[idx].to_dict()

                # 타임스탬프 갱신
                row['Timer'] = current_time.strftime('[%Y.%m.%d %H:%M:%S]')

                # 비정상 모드: PWPDS.Data 이상 주입
                if idx in anomaly_indices:
                    row = self._inject_pwpds_anomaly(row, recipe_key)

                # 비정상 모드: Ar 유량 진동 주입
                if idx in ar_fluctuation_indices:
                    row = self._inject_ar_fluctuation(row, idx, ar_fluctuation_indices)

                # 행 출력
                values = []
                for col in OUTPUT_COLUMNS:
                    val = row.get(col, 0)
                    if val is None:
                        val = 0
                    try:
                        fval = float(val)
                        if col == 'Timer':
                            values.append(str(val))
                        elif abs(fval) < 0.01 and fval != 0:
                            values.append(f'{fval:.2E}')
                        elif isinstance(val, float):
                            values.append(f'{fval}')
                        else:
                            values.append(str(val))
                    except (ValueError, TypeError):
                        values.append(str(val))

                f.write(','.join(values) + '\n')
                f.flush()

                time.sleep(1)

            print(f"[INFO] 공정 완료: {process_id} ({len(df)}행)")

        return process_id, out_path

    def _plan_anomalies(self, df: pd.DataFrame, recipe_key: str) -> set:
        """비정상 공정의 이상 주입 위치 결정.
        활성 공정 구간에서 1~3회, 각 1~3초 동안 이상 발생.
        """
        # 활성 구간 인덱스 찾기
        active_mask = (df['EN4.Power'] > 0) | (df['SBRF5.SetPower'] > 0)
        active_indices = df.index[active_mask].tolist()

        if len(active_indices) < 15:
            return set()

        # 1~3회 이상 발생
        n_events = random.randint(1, 3)
        anomaly_set = set()

        for _ in range(n_events):
            # 활성 구간 중간 부분에서 선택 (시작/끝 10% 제외)
            margin = max(1, len(active_indices) // 10)
            center = random.randint(margin, len(active_indices) - margin - 1)
            center_idx = active_indices[center]

            # 1~3초 지속
            duration = random.randint(1, 3)
            for d in range(duration):
                if center_idx + d < len(df):
                    anomaly_set.add(center_idx + d)

        return anomaly_set

    def _plan_ar_fluctuation(self, df: pd.DataFrame) -> set:
        """Ar 유량 진동 주입 위치 결정.
        활성 구간에서 4~8초 동안 ±1~2% 진동.
        """
        active_mask = (df['EN4.Power'] > 0) | (df['SBRF5.SetPower'] > 0)
        active_indices = df.index[active_mask].tolist()

        if len(active_indices) < 20:
            return set()

        # 50% 확률로 Ar 진동 이상 주입
        if random.random() > 0.5:
            return set()

        fluctuation_set = set()
        margin = max(1, len(active_indices) // 10)
        center = random.randint(margin, len(active_indices) - margin - 1)
        center_idx = active_indices[center]

        # 4~8초 지속
        duration = random.randint(4, 8)
        for d in range(duration):
            if center_idx + d < len(df):
                fluctuation_set.add(center_idx + d)

        return fluctuation_set

    def _inject_ar_fluctuation(self, row: dict, idx: int, fluctuation_indices: set) -> dict:
        """Ar.MFC.i에 진동 패턴 주입 (±1~2% oscillation)"""
        ar_val = float(row.get('Ar.MFC.i', 0))
        if ar_val < 1:
            return row

        # 교대로 +/- 패턴 (zero-crossing 생성)
        sorted_indices = sorted(fluctuation_indices)
        pos_in_seq = sorted_indices.index(idx) if idx in sorted_indices else 0
        amplitude = ar_val * random.uniform(0.012, 0.025)  # 1.2~2.5%
        sign = 1 if pos_in_seq % 2 == 0 else -1
        row['Ar.MFC.i'] = ar_val + sign * amplitude

        return row

    def _inject_pwpds_anomaly(self, row: dict, recipe_key: str) -> dict:
        """PWPDS.Data를 상/하한 밖으로 변경"""
        bounds = self.bounds.get(recipe_key)
        if not bounds:
            # bounds 없으면 큰 변동 추가
            row['PWPDS.Data'] = float(row.get('PWPDS.Data', 0)) * (1 + random.choice([-0.15, 0.15]))
            return row

        lower = bounds['lower']
        upper = bounds['upper']
        bound_range = upper - lower

        # 상한 초과 또는 하한 미만 (랜덤)
        if random.random() < 0.5:
            # 상한 초과 (10~30% 초과)
            overshoot = bound_range * random.uniform(0.1, 0.3)
            row['PWPDS.Data'] = upper + overshoot
        else:
            # 하한 미만 (10~30% 미달)
            undershoot = bound_range * random.uniform(0.1, 0.3)
            row['PWPDS.Data'] = lower - undershoot

        return row


def main():
    parser = argparse.ArgumentParser(description='PVD 공정 데이터 생성기 (PDS_Data_Log 기반)')
    parser.add_argument('--output-dir',
                        default='/home/goo4168/baco/pvd_monitoring/data',
                        help='출력 디렉토리')
    parser.add_argument('--inject-anomaly', action='store_true',
                        help='비정상 공정 (PWPDS.Data 이상 주입)')
    parser.add_argument('--continuous', action='store_true',
                        help='연속 실행')
    parser.add_argument('--count', type=int, default=1,
                        help='생성할 공정 수')

    args = parser.parse_args()

    generator = PVDDataGenerator(output_dir=args.output_dir)

    mode = 'abnormal' if args.inject_anomaly else 'normal'

    try:
        count = 0
        while True:
            process_id, filepath = generator.generate_process(mode=mode)
            count += 1
            print(f"[INFO] 생성된 공정 수: {count}")

            if not args.continuous and count >= args.count:
                break

            print("[INFO] 다음 공정까지 5초 대기...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[INFO] 데이터 생성 중단됨")
        sys.exit(0)


if __name__ == '__main__':
    main()
