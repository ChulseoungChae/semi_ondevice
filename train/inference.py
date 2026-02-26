#!/usr/bin/env python3
"""
PVD 예측 모델 추론 모듈
- 실시간 데이터 입력 시 예측값 출력
- 실제값과 비교하여 이상 탐지
"""

import os
import numpy as np
import torch
import joblib
from typing import Dict, List, Tuple, Optional
from collections import deque

# 모델 클래스 import
from pvd_predictor import LSTMPredictor, PatchTSTEncoder, Config


class PVDPredictor:
    """PVD 실시간 예측 및 이상 탐지 클래스"""

    def __init__(self, pvd_name: str, model_type: str = 'lstm',
                 device: str = 'cuda:0', threshold_multiplier: float = 3.0):
        """
        Args:
            pvd_name: PVD1, PVD2, PVD3, PVD4 중 하나
            model_type: 'lstm' 또는 'patchtst'
            device: 'cuda:0', 'cuda:1', 또는 'cpu'
            threshold_multiplier: 이상 탐지 임계값 배수 (MAE * multiplier)
        """
        self.pvd_name = pvd_name
        self.model_type = model_type
        self.config = Config()
        self.target_columns = Config.PVD_TARGETS[pvd_name]
        self.threshold_multiplier = threshold_multiplier

        # Device 설정
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 모델 및 스케일러 로드
        self._load_model()
        self._load_scalers()

        # 입력 버퍼 (sliding window)
        self.input_buffer = deque(maxlen=self.config.INPUT_WINDOW)

        # 이상 탐지 임계값 (학습 시 MAE 기반)
        self.thresholds = self._load_thresholds()

    def _load_model(self):
        """학습된 모델 로드"""
        model_dir = os.path.join(self.config.SAVE_PATH, self.pvd_name)
        model_path = os.path.join(model_dir, f'{self.model_type}_best.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # 입력/출력 크기 추론
        input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1] \
            if self.model_type == 'lstm' else \
            checkpoint['model_state_dict']['patch_embedding.weight'].shape[1] // 2
        output_size = len(self.target_columns)

        # 모델 생성 및 가중치 로드
        if self.model_type == 'lstm':
            self.model = LSTMPredictor(
                input_size=input_size,
                output_size=output_size,
                hidden_size=self.config.HIDDEN_SIZE,
                num_layers=self.config.NUM_LAYERS,
                dropout=0  # 추론 시 dropout 불필요
            )
        else:
            self.model = PatchTSTEncoder(
                input_size=input_size,
                output_size=output_size,
                d_model=self.config.HIDDEN_SIZE,
                dropout=0
            )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.input_size = input_size
        print(f"[{self.pvd_name}] Model loaded successfully (input_size={input_size})")

    def _load_scalers(self):
        """스케일러 로드"""
        model_dir = os.path.join(self.config.SAVE_PATH, self.pvd_name)
        self.scaler_input = joblib.load(os.path.join(model_dir, 'scaler_input.pkl'))
        self.scaler_target = joblib.load(os.path.join(model_dir, 'scaler_target.pkl'))
        print(f"[{self.pvd_name}] Scalers loaded successfully")

    def _load_thresholds(self) -> Dict[str, float]:
        """이상 탐지 임계값 로드"""
        model_dir = os.path.join(self.config.SAVE_PATH, self.pvd_name)
        results_path = os.path.join(model_dir, f'{self.model_type}_results.txt')

        thresholds = {}
        try:
            with open(results_path, 'r') as f:
                content = f.read()

            for col in self.target_columns:
                # MAE 값 파싱
                import re
                pattern = rf"{col}:.*?MAE:\s*([\d.]+)"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    mae = float(match.group(1))
                    thresholds[col] = mae * self.threshold_multiplier
                else:
                    thresholds[col] = 0.1  # 기본값
        except:
            # 파일이 없으면 기본값 사용
            for col in self.target_columns:
                thresholds[col] = 0.1

        print(f"[{self.pvd_name}] Anomaly thresholds: {thresholds}")
        return thresholds

    def add_data_point(self, data: np.ndarray) -> bool:
        """
        새로운 데이터 포인트 추가

        Args:
            data: 1D numpy array (모든 입력 특성)

        Returns:
            True if buffer is full and ready for prediction
        """
        self.input_buffer.append(data)
        return len(self.input_buffer) == self.config.INPUT_WINDOW

    def predict(self) -> Optional[Dict[str, float]]:
        """
        현재 버퍼 데이터로 예측 수행

        Returns:
            예측값 딕셔너리 {컬럼명: 예측값} 또는 버퍼가 부족하면 None
        """
        if len(self.input_buffer) < self.config.INPUT_WINDOW:
            return None

        # 입력 데이터 준비
        input_data = np.array(list(self.input_buffer))
        input_scaled = self.scaler_input.transform(input_data)

        # 텐서 변환
        input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(self.device)

        # 예측
        with torch.no_grad():
            output = self.model(input_tensor)
            output_np = output.cpu().numpy()

        # 역변환
        prediction = self.scaler_target.inverse_transform(output_np)[0]

        return {col: float(pred) for col, pred in zip(self.target_columns, prediction)}

    def detect_anomaly(self, prediction: Dict[str, float],
                       actual: Dict[str, float]) -> Dict[str, Dict]:
        """
        예측값과 실제값 비교하여 이상 탐지

        Args:
            prediction: 예측값 딕셔너리
            actual: 실제값 딕셔너리

        Returns:
            이상 탐지 결과 딕셔너리
        """
        results = {}

        for col in self.target_columns:
            if col not in actual:
                continue

            pred_val = prediction[col]
            actual_val = actual[col]
            error = abs(pred_val - actual_val)
            threshold = self.thresholds.get(col, 0.1)

            is_anomaly = error > threshold

            results[col] = {
                'predicted': pred_val,
                'actual': actual_val,
                'error': error,
                'threshold': threshold,
                'is_anomaly': is_anomaly
            }

        return results

    def reset_buffer(self):
        """입력 버퍼 초기화"""
        self.input_buffer.clear()


class AnomalyDetector:
    """다중 PVD 이상 탐지 통합 클래스"""

    def __init__(self, pvd_list: List[str] = None, model_type: str = 'lstm'):
        """
        Args:
            pvd_list: 모니터링할 PVD 목록 (기본: 전체)
            model_type: 'lstm' 또는 'patchtst'
        """
        if pvd_list is None:
            pvd_list = ['PVD1', 'PVD2', 'PVD3', 'PVD4']

        self.predictors = {}

        # GPU 할당
        gpu_map = {'PVD1': 0, 'PVD2': 0, 'PVD3': 1, 'PVD4': 1}

        for pvd_name in pvd_list:
            gpu_id = gpu_map.get(pvd_name, 0)
            device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            try:
                self.predictors[pvd_name] = PVDPredictor(
                    pvd_name, model_type, device
                )
            except Exception as e:
                print(f"Failed to load {pvd_name}: {e}")

    def process_data(self, pvd_name: str, data: np.ndarray,
                     actual_targets: Optional[Dict[str, float]] = None) -> Optional[Dict]:
        """
        데이터 처리 및 이상 탐지

        Args:
            pvd_name: PVD 이름
            data: 입력 데이터 (1D array)
            actual_targets: 실제 타겟 값 (5초 후 검증용)

        Returns:
            이상 탐지 결과 또는 None (버퍼 부족 시)
        """
        if pvd_name not in self.predictors:
            return None

        predictor = self.predictors[pvd_name]

        # 데이터 추가
        is_ready = predictor.add_data_point(data)

        if not is_ready:
            return None

        # 예측
        prediction = predictor.predict()

        if actual_targets is None:
            return {'prediction': prediction}

        # 이상 탐지
        anomaly_result = predictor.detect_anomaly(prediction, actual_targets)

        return {
            'prediction': prediction,
            'anomaly_detection': anomaly_result,
            'has_anomaly': any(r['is_anomaly'] for r in anomaly_result.values())
        }


# 테스트 및 데모
def demo():
    """데모 실행"""
    print("PVD Anomaly Detection Demo")
    print("="*50)

    # 단일 PVD 테스트
    try:
        predictor = PVDPredictor('PVD1', 'lstm', 'cuda:0')

        # 더미 데이터로 테스트
        input_size = predictor.input_size

        print(f"\nAdding {Config.INPUT_WINDOW} data points...")
        for i in range(Config.INPUT_WINDOW):
            dummy_data = np.random.randn(input_size)
            is_ready = predictor.add_data_point(dummy_data)
            print(f"  Point {i+1}: Buffer ready = {is_ready}")

        # 예측
        prediction = predictor.predict()
        print(f"\nPrediction (5 seconds ahead):")
        for col, val in prediction.items():
            print(f"  {col}: {val:.6f}")

        # 이상 탐지 테스트
        actual = {col: val * 1.01 for col, val in prediction.items()}  # 약간 다른 값
        anomaly_result = predictor.detect_anomaly(prediction, actual)

        print(f"\nAnomaly Detection Result:")
        for col, result in anomaly_result.items():
            status = "ANOMALY!" if result['is_anomaly'] else "Normal"
            print(f"  {col}: {status} (error={result['error']:.6f}, threshold={result['threshold']:.6f})")

    except FileNotFoundError as e:
        print(f"Model not found. Please train the model first.")
        print(f"Run: python pvd_predictor.py --pvd PVD1")


if __name__ == '__main__':
    demo()
