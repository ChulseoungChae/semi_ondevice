#!/usr/bin/env python3
"""
PVD4 LSTM 실시간 추론 엔진 (model_B_aug100x 기반)

입력: 10초 윈도우 x 5 features (Ar.MFC.i, EN4.Power, SBRF5.SetPower, PLA5.Match.DCBias, PWPDS.Data)
출력: 5초 후 PWPDS.Data 예측값
"""

import os
import sys
import json
import re
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

# 모델 설정 (model_B_aug100x)
MODEL_NAME = 'model_B_aug100x'
INPUT_COLUMNS = ['Ar.MFC.i', 'EN4.Power', 'SBRF5.SetPower', 'PLA5.Match.DCBias', 'PWPDS.Data']
OUTPUT_COLUMNS = ['PWPDS.Data']
INPUT_WINDOW = 10
PREDICTION_HORIZON = 5
ANOMALY_THRESHOLD_PCT = 10.0  # 10% 차이 시 이상


class PVDLSTMModel(nn.Module):
    """LSTM 모델 (models.py와 동일한 구조)"""

    def __init__(self, n_features: int, n_outputs: int = 1,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_outputs),
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.output_layer(context)


def detect_recipe(data: Dict) -> dict:
    """현재 데이터에서 레시피 감지"""
    dc = float(data.get('EN4.Power', 0))
    rf = float(data.get('SBRF5.SetPower', 0))

    # DC 설정값 추정 (실제 값은 약간 변동)
    dc_settings = [0, 1000, 2000, 3000]
    dc_setting = min(dc_settings, key=lambda x: abs(x - dc))

    # RF 설정값 추정
    rf_settings = [0, 300, 400, 500]
    rf_setting = min(rf_settings, key=lambda x: abs(x - rf))

    return {
        'dc_setting': dc_setting,
        'rf_setting': rf_setting,
        'dc_actual': dc,
        'rf_actual': rf,
        'recipe_key': f"DC{dc_setting}_RF{rf_setting}"
    }


class PVD4InferenceEngine:
    """PVD4 실시간 추론 엔진 (PWPDS.Data 전용)"""

    TARGET_COLUMNS = OUTPUT_COLUMNS

    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.bounds = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_buffer = deque(maxlen=INPUT_WINDOW)
        self.prediction_queue = deque()
        self.data_index = 0
        self.predictions_history = []
        self.actuals_history = []
        self.anomaly_logs = []
        self.current_anomaly_intervals = {}
        self.current_recipe = {'dc_setting': 0, 'rf_setting': 0, 'recipe_key': 'DC0_RF0'}
        self.lock = threading.Lock()
        self.loaded = False

    def load_model(self) -> bool:
        """모델, 스케일러, 바운드 로드"""
        try:
            model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_best.pt')
            scaler_x_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_scaler_x.pkl')
            scaler_y_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_scaler_y.pkl')
            bounds_path = os.path.join(RESULT_DIR, 'bounds_lookup.json')

            if not os.path.exists(model_path):
                print(f"[InferenceEngine] 모델 파일 없음: {model_path}")
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

            # 모델 로드
            n_features = len(INPUT_COLUMNS)
            n_outputs = len(OUTPUT_COLUMNS)
            self.model = PVDLSTMModel(n_features=n_features, n_outputs=n_outputs)

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.loaded = True
            print(f"[InferenceEngine] 모델 로드 완료: {MODEL_NAME}")
            print(f"[InferenceEngine] Device: {self.device}, Features: {n_features}")
            print(f"[InferenceEngine] 바운드: {len(self.bounds)} 레시피")
            return True

        except Exception as e:
            print(f"[InferenceEngine] 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, data: Dict, timestamp: str = None) -> Optional[Dict]:
        """실시간 예측 수행"""
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
                for col in INPUT_COLUMNS:
                    val = data.get(col, 0)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = 0
                    values.append(float(val))
                processed = np.array(values, dtype=np.float32)

                # 버퍼에 추가
                self.input_buffer.append(processed)

                if len(self.input_buffer) < INPUT_WINDOW:
                    return None

                # 입력 시퀀스 생성 및 스케일링
                input_seq = np.array(list(self.input_buffer), dtype=np.float64)
                n_feat = input_seq.shape[1]
                input_scaled = self.scaler_x.transform(input_seq.reshape(-1, n_feat)).reshape(1, INPUT_WINDOW, n_feat)

                input_tensor = torch.FloatTensor(input_scaled).to(self.device)

                # 추론
                with torch.no_grad():
                    output = self.model(input_tensor)
                    output_np = output.cpu().numpy()

                # 역스케일링
                pred_original = self.scaler_y.inverse_transform(output_np)[0]

                # 현재 실제 PWPDS.Data
                actual_pwpds = float(data.get('PWPDS.Data', 0))

                # 예측값 (5초 후)
                pred_pwpds = float(pred_original[0])

                ts = timestamp or datetime.now().isoformat()
                current_idx = self.data_index
                target_idx = current_idx + PREDICTION_HORIZON
                self.data_index += 1

                # 예측 대기열에 저장
                self.prediction_queue.append({
                    'target_index': target_idx,
                    'pred_pwpds': pred_pwpds,
                })

                # 현재 실제값 저장
                self.actuals_history.append({
                    'timestamp': ts,
                    'index': current_idx,
                    'actual': actual_pwpds,
                })

                # 예측값 기록 (target_idx 기준)
                self.predictions_history.append({
                    'timestamp': ts,
                    'index': target_idx,
                    'predicted': pred_pwpds,
                })

                # 최대 1000개 유지
                if len(self.predictions_history) > 1000:
                    self.predictions_history = self.predictions_history[-1000:]
                    self.actuals_history = self.actuals_history[-1000:]

                # 5초 전 예측과 현재 실제값 비교
                error_pct = 0.0
                matched_pred = None
                is_anomaly_prediction = False

                while self.prediction_queue:
                    pending = self.prediction_queue[0]
                    if pending['target_index'] <= current_idx:
                        matched_pred = self.prediction_queue.popleft()
                    else:
                        break

                if matched_pred:
                    if abs(actual_pwpds) > 1e-6:
                        error_pct = abs(matched_pred['pred_pwpds'] - actual_pwpds) / abs(actual_pwpds) * 100
                    is_anomaly_prediction = error_pct >= ANOMALY_THRESHOLD_PCT

                    if is_anomaly_prediction:
                        self._update_anomaly_interval('PREDICTION_ERROR', ts, error_pct)
                    else:
                        self._close_anomaly_interval('PREDICTION_ERROR', ts)

                # 상/하한 체크
                bounds_info = self.bounds.get(recipe['recipe_key'], {})
                lower_bound = bounds_info.get('lower', None)
                upper_bound = bounds_info.get('upper', None)

                is_out_of_bounds = False
                if lower_bound is not None and upper_bound is not None:
                    is_out_of_bounds = actual_pwpds < lower_bound or actual_pwpds > upper_bound
                    if is_out_of_bounds:
                        self._update_anomaly_interval('BOUNDS_VIOLATION', ts, 0)
                    else:
                        self._close_anomaly_interval('BOUNDS_VIOLATION', ts)

                result = {
                    'timestamp': ts,
                    'index': current_idx,
                    'actual_pwpds': actual_pwpds,
                    'predicted_pwpds': pred_pwpds,
                    'error_pct': error_pct,
                    'is_anomaly_prediction': is_anomaly_prediction,
                    'is_out_of_bounds': is_out_of_bounds,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'recipe': recipe,
                    'sensor_data': {
                        'Ar.MFC.i': float(data.get('Ar.MFC.i', 0)),
                        'EN4.Power': float(data.get('EN4.Power', 0)),
                        'SBRF5.SetPower': float(data.get('SBRF5.SetPower', 0)),
                        'PLA5.Match.DCBias': float(data.get('PLA5.Match.DCBias', 0)),
                        'PWPDS.Data': actual_pwpds,
                    },
                    'has_comparison': matched_pred is not None,
                }

                return result

            except Exception as e:
                print(f"[InferenceEngine] 예측 오류: {e}")
                import traceback
                traceback.print_exc()
                return None

    def _update_anomaly_interval(self, atype: str, timestamp: str, error_pct: float):
        if atype not in self.current_anomaly_intervals:
            self.current_anomaly_intervals[atype] = {
                'type': atype,
                'start_time': timestamp,
                'end_time': timestamp,
                'max_error_pct': error_pct,
                'count': 1,
            }
        else:
            inv = self.current_anomaly_intervals[atype]
            inv['end_time'] = timestamp
            inv['max_error_pct'] = max(inv['max_error_pct'], error_pct)
            inv['count'] += 1

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
                'start_time': inv['start_time'],
                'end_time': inv['end_time'],
                'duration_sec': duration_sec,
                'max_error_pct': inv['max_error_pct'],
                'data_points': inv['count'],
                'closed_at': timestamp,
                'status': 'closed',
            })

            if len(self.anomaly_logs) > 500:
                self.anomaly_logs = self.anomaly_logs[:500]

            del self.current_anomaly_intervals[atype]

    def get_chart_data(self, count: int = 200) -> Dict:
        with self.lock:
            return {
                'predictions': self.predictions_history[-count:],
                'actuals': self.actuals_history[-count:],
                'columns': self.TARGET_COLUMNS,
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
                    'start_time': inv['start_time'],
                    'end_time': '진행중',
                    'duration_sec': duration_sec,
                    'max_error_pct': inv['max_error_pct'],
                    'data_points': inv['count'],
                    'status': 'ongoing',
                })
            closed = self.anomaly_logs[:limit - len(ongoing)]
            return ongoing + closed

    def get_status(self) -> Dict:
        return {
            'loaded': self.loaded,
            'device': str(self.device),
            'model_name': MODEL_NAME,
            'buffer_size': len(self.input_buffer),
            'buffer_required': INPUT_WINDOW,
            'predictions_count': len(self.predictions_history),
            'anomaly_logs_count': len(self.anomaly_logs),
            'ongoing_anomalies': len(self.current_anomaly_intervals),
            'target_columns': self.TARGET_COLUMNS,
            'recipe': self.current_recipe,
        }

    def reset(self):
        with self.lock:
            self.input_buffer.clear()
            self.prediction_queue.clear()
            self.data_index = 0
            self.predictions_history.clear()
            self.actuals_history.clear()
            self.current_anomaly_intervals.clear()


# 싱글톤
_inference_engine: Optional[PVD4InferenceEngine] = None

def get_inference_engine() -> PVD4InferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = PVD4InferenceEngine()
    return _inference_engine
