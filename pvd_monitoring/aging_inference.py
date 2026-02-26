#!/usr/bin/env python3
"""
장비 노후 감지 추론 엔진 (lstm_7feat 모델)

입력: 10초 윈도우 x 7 features
출력: 7개 타겟 5초 후 예측
  Ar.MFC.i, Baratron.Gauge.i, PLA5.Match.DCBias, EN4.Power,
  SBRF5.SetPower, PWPDS.Data, Ion.Gauge.i
"""

import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import threading


# 모델 경로
MODEL_DIR = '/home/goo4168/baco/train/models/PVD4'
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_7feat_best.pth')
SCALER_INPUT_PATH  = os.path.join(MODEL_DIR, 'scaler_input_lstm_7feat.pkl')
SCALER_TARGET_PATH = os.path.join(MODEL_DIR, 'scaler_target_lstm_7feat.pkl')

# 7개 입력/타겟 칼럼
FEATURES = [
    'Ar.MFC.i', 'Baratron.Gauge.i', 'PLA5.Match.DCBias',
    'EN4.Power', 'SBRF5.SetPower', 'PWPDS.Data', 'Ion.Gauge.i',
]

INPUT_WINDOW = 10
PREDICTION_HORIZON = 5
ANOMALY_THRESHOLD_PCT = 10.0


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
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
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


class AgingInferenceEngine:
    """장비 노후 감지 추론 엔진 (7개 칼럼 동시 예측)"""

    def __init__(self):
        self.model = None
        self.scaler_input = None
        self.scaler_target = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_buffer = deque(maxlen=INPUT_WINDOW)
        self.prediction_queue = deque()
        self.data_index = 0

        self.actuals_history: List[Dict] = []
        self.predictions_history: List[Dict] = []
        self.anomaly_logs: List[Dict] = []
        self.current_anomaly_intervals: Dict[str, Dict] = {}

        self.lock = threading.Lock()
        self.loaded = False

    def load_model(self) -> bool:
        try:
            if not os.path.exists(MODEL_PATH):
                print(f"[AgingEngine] 모델 파일 없음: {MODEL_PATH}")
                return False

            self.scaler_input  = joblib.load(SCALER_INPUT_PATH)
            self.scaler_target = joblib.load(SCALER_TARGET_PATH)

            self.model = LSTMPredictor(
                input_size=len(FEATURES),
                output_size=len(FEATURES),
            )
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.loaded = True
            print(f"[AgingEngine] 모델 로드 완료 (device={self.device})")
            return True
        except Exception as e:
            print(f"[AgingEngine] 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, data: Dict, timestamp: str = None) -> Optional[Dict]:
        if not self.loaded:
            if not self.load_model():
                return None

        with self.lock:
            try:
                # 7개 입력 피처 추출
                values = []
                for col in FEATURES:
                    val = data.get(col, 0)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = 0
                    values.append(float(val))
                processed = np.array(values, dtype=np.float32)

                self.input_buffer.append(processed)
                if len(self.input_buffer) < INPUT_WINDOW:
                    return None

                # 스케일링 및 추론
                input_seq = np.array(list(self.input_buffer), dtype=np.float64)
                n_feat = input_seq.shape[1]
                input_scaled = self.scaler_input.transform(
                    input_seq.reshape(-1, n_feat)
                ).reshape(1, INPUT_WINDOW, n_feat)

                input_tensor = torch.FloatTensor(input_scaled).to(self.device)

                with torch.no_grad():
                    output = self.model(input_tensor)
                    output_np = output.cpu().numpy()

                pred_original = self.scaler_target.inverse_transform(output_np)[0]

                ts = timestamp or datetime.now().isoformat()
                current_idx = self.data_index
                target_idx  = current_idx + PREDICTION_HORIZON
                self.data_index += 1

                # 실제값
                actuals = {col: float(data.get(col, 0) or 0) for col in FEATURES}

                # 예측값
                preds = {col: float(pred_original[i]) for i, col in enumerate(FEATURES)}

                # 예측 대기열
                self.prediction_queue.append({
                    'target_index': target_idx,
                    'predictions': preds.copy(),
                })

                self.actuals_history.append({'timestamp': ts, 'index': current_idx, **actuals})
                self.predictions_history.append({'timestamp': ts, 'index': target_idx, **preds})

                if len(self.predictions_history) > 1000:
                    self.predictions_history = self.predictions_history[-1000:]
                    self.actuals_history     = self.actuals_history[-1000:]

                # 5초 전 예측과 현재 실제값 비교
                matched_pred = None
                while self.prediction_queue:
                    pending = self.prediction_queue[0]
                    if pending['target_index'] <= current_idx:
                        matched_pred = self.prediction_queue.popleft()
                    else:
                        break

                if matched_pred:
                    for col in FEATURES:
                        actual_val = actuals[col]
                        pred_val   = matched_pred['predictions'][col]
                        if abs(actual_val) > 1e-6:
                            error_pct = abs(pred_val - actual_val) / abs(actual_val) * 100
                        else:
                            error_pct = 0.0

                        if error_pct >= ANOMALY_THRESHOLD_PCT:
                            self._update_anomaly_interval(col, ts, error_pct)
                        else:
                            self._close_anomaly_interval(col, ts)

                return {
                    'timestamp': ts,
                    'index': current_idx,
                    'actuals': actuals,
                    'predictions': preds,
                }

            except Exception as e:
                print(f"[AgingEngine] 예측 오류: {e}")
                import traceback
                traceback.print_exc()
                return None

    def _update_anomaly_interval(self, col: str, timestamp: str, error_pct: float):
        key = f"AGING_{col}"
        if key not in self.current_anomaly_intervals:
            self.current_anomaly_intervals[key] = {
                'type': key, 'column': col,
                'start_time': timestamp, 'end_time': timestamp,
                'max_error_pct': error_pct, 'count': 1,
            }
        else:
            inv = self.current_anomaly_intervals[key]
            inv['end_time'] = timestamp
            inv['max_error_pct'] = max(inv['max_error_pct'], error_pct)
            inv['count'] += 1

    def _close_anomaly_interval(self, col: str, timestamp: str):
        key = f"AGING_{col}"
        if key not in self.current_anomaly_intervals:
            return
        inv = self.current_anomaly_intervals[key]
        try:
            duration_sec = (
                datetime.fromisoformat(inv['end_time']) -
                datetime.fromisoformat(inv['start_time'])
            ).total_seconds()
        except Exception:
            duration_sec = 0

        self.anomaly_logs.insert(0, {
            'type': inv['type'], 'column': inv['column'],
            'start_time': inv['start_time'], 'end_time': inv['end_time'],
            'duration_sec': duration_sec, 'max_error_pct': inv['max_error_pct'],
            'data_points': inv['count'], 'closed_at': timestamp, 'status': 'closed',
        })
        if len(self.anomaly_logs) > 500:
            self.anomaly_logs = self.anomaly_logs[:500]
        del self.current_anomaly_intervals[key]

    def get_chart_data(self, count: int = 200) -> Dict:
        with self.lock:
            return {
                'columns': FEATURES,
                'actuals': self.actuals_history[-count:],
                'predictions': self.predictions_history[-count:],
            }

    def get_anomaly_logs(self, limit: int = 100) -> List[Dict]:
        with self.lock:
            ongoing = []
            for key, inv in self.current_anomaly_intervals.items():
                try:
                    duration_sec = (
                        datetime.now() - datetime.fromisoformat(inv['start_time'])
                    ).total_seconds()
                except Exception:
                    duration_sec = 0
                ongoing.append({
                    'type': inv['type'], 'column': inv['column'],
                    'start_time': inv['start_time'], 'end_time': '진행중',
                    'duration_sec': duration_sec, 'max_error_pct': inv['max_error_pct'],
                    'data_points': inv['count'], 'status': 'ongoing',
                })
            closed = self.anomaly_logs[:limit - len(ongoing)]
            return ongoing + closed

    def get_status(self) -> Dict:
        return {
            'loaded': self.loaded,
            'device': str(self.device),
            'buffer_size': len(self.input_buffer),
            'buffer_required': INPUT_WINDOW,
            'predictions_count': len(self.predictions_history),
            'anomaly_logs_count': len(self.anomaly_logs),
            'ongoing_anomalies': len(self.current_anomaly_intervals),
            'target_columns': FEATURES,
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
_aging_engine: Optional[AgingInferenceEngine] = None


def get_aging_engine() -> AgingInferenceEngine:
    global _aging_engine
    if _aging_engine is None:
        _aging_engine = AgingInferenceEngine()
    return _aging_engine
