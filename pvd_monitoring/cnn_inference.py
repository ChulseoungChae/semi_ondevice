#!/usr/bin/env python3
"""
PVD4 CNN 실시간 추론 엔진 (v3_CNN_alldata_aug50x 기반)

입력: 10초 윈도우 x 8 features
출력: 현재 시점 PWPDS.Data 예측값 (predict_ahead=0)
"""

import os
import sys
import json
import csv
import numpy as np
import pickle
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import threading

# LSTM 프로젝트 경로
LSTM_PROJECT_DIR = '/home/goo4168/baco/lstm_project'
MODEL_DIR = os.path.join(LSTM_PROJECT_DIR, 'models')
RESULT_DIR = os.path.join(LSTM_PROJECT_DIR, 'results')

# 모델 설정 (v3_CNN_alldata_aug50x)
CNN_MODEL_NAME = 'v3_CNN_alldata_aug50x'
CNN_INPUT_COLUMNS = [
    'SBRF5.SetPower',
    'EN4.Power',
    'Ar.MFC.i',
    'PLA5.Match.Tune.Posi',
    'PLA5.Match.DCBias',
    'ULVAC.Stage1.Temp1',
    'EN4.Volt',
    'Ion.Gauge.i',
]
CNN_OUTPUT_COLUMNS = ['PWPDS.Data']
CNN_INPUT_WINDOW = 10
CNN_PREDICT_AHEAD = 0  # 현재 시점 예측

# 이상 임계값
CNN_WARNING_THRESHOLD_PCT = 5.0   # 5% 차이 = 경고 (주황)
CNN_CRITICAL_THRESHOLD_PCT = 10.0  # 10% 차이 = 심각 (빨강)

# CSV Export 디렉토리
EXPORT_DIR = '/home/goo4168/baco/pvd_monitoring/exports'


class PVD1DCNNModel(nn.Module):
    """1D CNN model (models_v2.py 동일 구조)"""

    def __init__(self, n_features: int, n_outputs: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.conv_layers(x)  # (batch, 128, 1)
        x = x.squeeze(-1)        # (batch, 128)
        x = self.fc(x)           # (batch, n_outputs)
        return x


# 원인분석 매핑
CAUSE_ANALYSIS = {
    'actual_below': {
        'cause': '마그넷 오염',
        'detail': '금속입자/산화물/잔류타겟 → 자기장↓ → 전자궤도유지력↓ → 플라즈마밀도↓ → 스퍼터링속도↓',
        'impact': 'PWPDS.Data(플라즈마 밀도) 감소로 증착 균일도 저하',
        'recommendation': '마그넷 상태 점검 및 클리닝, 타겟 교체 검토',
    },
    'actual_above': {
        'cause': '타겟-웨이퍼 거리 감소',
        'detail': '타겟 근처 고밀도 플라즈마 → 거리↓ → 밀도↑ → edge 증착 불균일',
        'impact': 'PWPDS.Data(플라즈마 밀도) 증가로 edge 증착 불균일',
        'recommendation': '타겟-웨이퍼 간격 점검, 셔터 동작 확인',
    },
}


def detect_recipe(data: Dict) -> dict:
    """현재 데이터에서 레시피 감지"""
    dc = float(data.get('EN4.Power', 0))
    rf = float(data.get('SBRF5.SetPower', 0))

    dc_settings = [0, 1000, 2000, 3000]
    dc_setting = min(dc_settings, key=lambda x: abs(x - dc))
    rf_settings = [0, 300, 400, 500]
    rf_setting = min(rf_settings, key=lambda x: abs(x - rf))

    return {
        'dc_setting': dc_setting,
        'rf_setting': rf_setting,
        'dc_actual': dc,
        'rf_actual': rf,
        'recipe_key': f"DC{dc_setting}_RF{rf_setting}"
    }


class CNNInferenceEngine:
    """PVD4 CNN 실시간 추론 엔진 (현재 시점 PWPDS.Data 예측)"""

    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.bounds = {}
        self.column_ranges = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_buffer = deque(maxlen=CNN_INPUT_WINDOW)
        self.data_index = 0
        self.predictions_history = []
        self.actuals_history = []
        self.anomaly_logs = []
        self.current_anomaly_intervals = {}
        self.current_recipe = {'dc_setting': 0, 'rf_setting': 0, 'recipe_key': 'DC0_RF0'}
        self.lock = threading.Lock()
        self.loaded = False

        # CSV export
        self.csv_writer = None
        self.csv_file = None
        self.csv_filepath = None
        self.export_dir = EXPORT_DIR
        os.makedirs(self.export_dir, exist_ok=True)

    def load_model(self) -> bool:
        """모델, 스케일러, 바운드, column_ranges 로드"""
        try:
            model_path = os.path.join(MODEL_DIR, f'{CNN_MODEL_NAME}_best.pt')
            scaler_x_path = os.path.join(MODEL_DIR, f'{CNN_MODEL_NAME}_scaler_x.pkl')
            scaler_y_path = os.path.join(MODEL_DIR, f'{CNN_MODEL_NAME}_scaler_y.pkl')
            bounds_path = os.path.join(RESULT_DIR, 'bounds_lookup.json')
            ranges_path = os.path.join(RESULT_DIR, 'column_ranges.json')

            if not os.path.exists(model_path):
                print(f"[CNNEngine] 모델 파일 없음: {model_path}")
                return False

            # 스케일러 로드
            with open(scaler_x_path, 'rb') as f:
                self.scaler_x = pickle.load(f)
            with open(scaler_y_path, 'rb') as f:
                self.scaler_y = pickle.load(f)

            # 바운드 로드
            if os.path.exists(bounds_path):
                with open(bounds_path, 'r') as f:
                    self.bounds = json.load(f)

            # column_ranges 로드
            if os.path.exists(ranges_path):
                with open(ranges_path, 'r') as f:
                    self.column_ranges = json.load(f)

            # 모델 로드
            n_features = len(CNN_INPUT_COLUMNS)
            n_outputs = len(CNN_OUTPUT_COLUMNS)
            self.model = PVD1DCNNModel(n_features=n_features, n_outputs=n_outputs)

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.loaded = True
            print(f"[CNNEngine] 모델 로드 완료: {CNN_MODEL_NAME}")
            print(f"[CNNEngine] Device: {self.device}, Features: {n_features}")
            print(f"[CNNEngine] 바운드: {len(self.bounds)} 레시피")
            return True

        except Exception as e:
            print(f"[CNNEngine] 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, data: Dict, timestamp: str = None) -> Optional[Dict]:
        """실시간 CNN 예측 수행 (현재 시점)"""
        if not self.loaded:
            if not self.load_model():
                return None

        with self.lock:
            try:
                # 레시피 감지
                recipe = detect_recipe(data)
                self.current_recipe = recipe

                # 입력 피처 추출
                values = []
                for col in CNN_INPUT_COLUMNS:
                    val = data.get(col, 0)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = 0
                    values.append(float(val))
                processed = np.array(values, dtype=np.float32)

                # 버퍼에 추가
                self.input_buffer.append(processed)

                if len(self.input_buffer) < CNN_INPUT_WINDOW:
                    return None

                # 입력 시퀀스 생성 및 스케일링
                input_seq = np.array(list(self.input_buffer), dtype=np.float64)
                n_feat = input_seq.shape[1]
                input_scaled = self.scaler_x.transform(
                    input_seq.reshape(-1, n_feat)
                ).reshape(1, CNN_INPUT_WINDOW, n_feat)

                input_tensor = torch.FloatTensor(input_scaled).to(self.device)

                # 추론
                with torch.no_grad():
                    output = self.model(input_tensor)
                    output_np = output.cpu().numpy()

                # 역스케일링
                pred_original = self.scaler_y.inverse_transform(output_np)[0]
                pred_pwpds = float(pred_original[0])

                # 현재 실제 PWPDS.Data
                actual_pwpds = float(data.get('PWPDS.Data', 0))

                ts = timestamp or datetime.now().isoformat()
                current_idx = self.data_index
                self.data_index += 1

                # 예측 오차 계산 (현재 시점이므로 바로 비교)
                error_pct = 0.0
                if abs(actual_pwpds) > 1e-6:
                    error_pct = abs(pred_pwpds - actual_pwpds) / abs(actual_pwpds) * 100

                # 이상 판단
                severity = None
                anomaly_type = None
                cause_info = None

                if error_pct >= CNN_CRITICAL_THRESHOLD_PCT:
                    severity = 'critical'
                    anomaly_type = 'PWPDS_PREDICTION_ANOMALY'
                elif error_pct >= CNN_WARNING_THRESHOLD_PCT:
                    severity = 'warning'
                    anomaly_type = 'PWPDS_PREDICTION_ANOMALY'

                # 상/하한 체크
                bounds_info = self.bounds.get(recipe['recipe_key'], {})
                lower_bound = bounds_info.get('lower', None)
                upper_bound = bounds_info.get('upper', None)
                is_out_of_bounds = False

                if lower_bound is not None and upper_bound is not None:
                    if actual_pwpds < lower_bound:
                        is_out_of_bounds = True
                        severity = 'critical'
                        anomaly_type = 'PWPDS_PREDICTION_ANOMALY'
                    elif actual_pwpds > upper_bound:
                        is_out_of_bounds = True
                        severity = 'critical'
                        anomaly_type = 'PWPDS_PREDICTION_ANOMALY'

                # 원인분석
                if anomaly_type:
                    if actual_pwpds < pred_pwpds or (lower_bound and actual_pwpds < lower_bound):
                        cause_info = CAUSE_ANALYSIS['actual_below']
                    else:
                        cause_info = CAUSE_ANALYSIS['actual_above']

                    self._update_anomaly_interval(anomaly_type, severity, ts, error_pct, cause_info)
                else:
                    self._close_anomaly_interval('PWPDS_PREDICTION_ANOMALY', ts)

                # 히스토리 저장
                self.actuals_history.append({
                    'timestamp': ts,
                    'index': current_idx,
                    'actual': actual_pwpds,
                })
                self.predictions_history.append({
                    'timestamp': ts,
                    'index': current_idx,
                    'predicted': pred_pwpds,
                })

                if len(self.predictions_history) > 1000:
                    self.predictions_history = self.predictions_history[-1000:]
                    self.actuals_history = self.actuals_history[-1000:]

                # 센서 데이터 수집
                sensor_data = {}
                for col in CNN_INPUT_COLUMNS:
                    sensor_data[col] = float(data.get(col, 0))
                sensor_data['PWPDS.Data'] = actual_pwpds

                result = {
                    'timestamp': ts,
                    'index': current_idx,
                    'actual_pwpds': actual_pwpds,
                    'predicted_pwpds': pred_pwpds,
                    'error_pct': error_pct,
                    'severity': severity,
                    'anomaly_type': anomaly_type,
                    'cause_info': cause_info,
                    'is_out_of_bounds': is_out_of_bounds,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'recipe': recipe,
                    'sensor_data': sensor_data,
                }

                # CSV에 기록
                self._write_csv_row(result, data)

                return result

            except Exception as e:
                print(f"[CNNEngine] 예측 오류: {e}")
                import traceback
                traceback.print_exc()
                return None

    def _update_anomaly_interval(self, atype: str, severity: str, timestamp: str,
                                  error_pct: float, cause_info: Optional[Dict]):
        if atype not in self.current_anomaly_intervals:
            self.current_anomaly_intervals[atype] = {
                'type': atype,
                'severity': severity,
                'start_time': timestamp,
                'end_time': timestamp,
                'max_error_pct': error_pct,
                'count': 1,
                'cause_info': cause_info,
            }
        else:
            inv = self.current_anomaly_intervals[atype]
            inv['end_time'] = timestamp
            inv['max_error_pct'] = max(inv['max_error_pct'], error_pct)
            inv['count'] += 1
            if severity == 'critical':
                inv['severity'] = 'critical'
            if cause_info:
                inv['cause_info'] = cause_info

    def _close_anomaly_interval(self, atype: str, timestamp: str):
        if atype in self.current_anomaly_intervals:
            inv = self.current_anomaly_intervals[atype]
            try:
                start = datetime.fromisoformat(inv['start_time'])
                end = datetime.fromisoformat(inv['end_time'])
                duration_sec = (end - start).total_seconds()
            except:
                duration_sec = 0

            self.anomaly_logs.insert(0, {
                'type': inv['type'],
                'severity': inv['severity'],
                'start_time': inv['start_time'],
                'end_time': inv['end_time'],
                'duration_sec': duration_sec,
                'max_error_pct': inv['max_error_pct'],
                'data_points': inv['count'],
                'cause_info': inv.get('cause_info'),
                'closed_at': timestamp,
                'status': 'closed',
            })

            if len(self.anomaly_logs) > 500:
                self.anomaly_logs = self.anomaly_logs[:500]

            del self.current_anomaly_intervals[atype]

    def start_csv_export(self, process_id: str):
        """CSV export 시작"""
        self.csv_filepath = os.path.join(self.export_dir, f'cnn_pred_{process_id}.csv')
        self.csv_file = open(self.csv_filepath, 'w', newline='', encoding='utf-8')
        fieldnames = [
            'timestamp', 'index',
            'actual_pwpds', 'predicted_pwpds', 'error_pct',
            'severity', 'anomaly_type', 'cause',
            'lower_bound', 'upper_bound', 'is_out_of_bounds',
            'recipe_key',
        ] + CNN_INPUT_COLUMNS
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        print(f"[CNNEngine] CSV export 시작: {self.csv_filepath}")

    def _write_csv_row(self, result: Dict, raw_data: Dict):
        """CSV에 데이터포인트 기록"""
        if self.csv_writer is None:
            return
        try:
            row = {
                'timestamp': result['timestamp'],
                'index': result['index'],
                'actual_pwpds': result['actual_pwpds'],
                'predicted_pwpds': result['predicted_pwpds'],
                'error_pct': result['error_pct'],
                'severity': result.get('severity', ''),
                'anomaly_type': result.get('anomaly_type', ''),
                'cause': result['cause_info']['cause'] if result.get('cause_info') else '',
                'lower_bound': result.get('lower_bound', ''),
                'upper_bound': result.get('upper_bound', ''),
                'is_out_of_bounds': result.get('is_out_of_bounds', False),
                'recipe_key': result['recipe']['recipe_key'],
            }
            for col in CNN_INPUT_COLUMNS:
                row[col] = result['sensor_data'].get(col, 0)
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            print(f"[CNNEngine] CSV 쓰기 오류: {e}")

    def finish_csv_export(self):
        """CSV export 완료"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"[CNNEngine] CSV export 완료: {self.csv_filepath}")

    def get_chart_data(self, count: int = 200) -> Dict:
        with self.lock:
            return {
                'predictions': self.predictions_history[-count:],
                'actuals': self.actuals_history[-count:],
                'columns': CNN_OUTPUT_COLUMNS,
                'recipe': self.current_recipe,
                'bounds': self.bounds.get(self.current_recipe.get('recipe_key', ''), {}),
            }

    def get_anomaly_logs(self, limit: int = 100) -> List[Dict]:
        with self.lock:
            ongoing = []
            for atype, inv in self.current_anomaly_intervals.items():
                try:
                    start = datetime.fromisoformat(inv['start_time'])
                    duration_sec = (datetime.now() - start).total_seconds()
                except:
                    duration_sec = 0
                ongoing.append({
                    'type': inv['type'],
                    'severity': inv['severity'],
                    'start_time': inv['start_time'],
                    'end_time': '진행중',
                    'duration_sec': duration_sec,
                    'max_error_pct': inv['max_error_pct'],
                    'data_points': inv['count'],
                    'cause_info': inv.get('cause_info'),
                    'status': 'ongoing',
                })
            closed = self.anomaly_logs[:limit - len(ongoing)]
            return ongoing + closed

    def get_column_ranges(self) -> Dict:
        """column_ranges.json 반환"""
        return self.column_ranges

    def get_status(self) -> Dict:
        return {
            'loaded': self.loaded,
            'device': str(self.device),
            'model_name': CNN_MODEL_NAME,
            'buffer_size': len(self.input_buffer),
            'buffer_required': CNN_INPUT_WINDOW,
            'predictions_count': len(self.predictions_history),
            'anomaly_logs_count': len(self.anomaly_logs),
            'ongoing_anomalies': len(self.current_anomaly_intervals),
            'input_columns': CNN_INPUT_COLUMNS,
            'recipe': self.current_recipe,
        }

    def reset(self):
        with self.lock:
            self.input_buffer.clear()
            self.data_index = 0
            self.predictions_history.clear()
            self.actuals_history.clear()
            self.current_anomaly_intervals.clear()
            self.finish_csv_export()


# 싱글톤
_cnn_engine: Optional[CNNInferenceEngine] = None

def get_cnn_engine() -> CNNInferenceEngine:
    global _cnn_engine
    if _cnn_engine is None:
        _cnn_engine = CNNInferenceEngine()
    return _cnn_engine
