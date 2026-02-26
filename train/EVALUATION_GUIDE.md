# PVD4 모델 검증 가이드

## 개요
학습된 LSTM 예측 모델의 성능을 검증하고 평가하는 가이드입니다.

---

## 파일 구조

```
train/
├── evaluate_pvd4_models.py    # 모델 평가/검증
├── visualize_prediction.py    # 예측 결과 시각화
└── models/
    └── PVD4/
        ├── lstm_aug100_extended_best.pth
        ├── lstm_aug50_best.pth
        ├── patchtst_aug50_best.pth
        └── comparison_aug50_results.txt
```

---

## 1. 모델 평가 (LSTM vs PatchTST)

### 실행 명령
```bash
cd /home/goo4168/baco/train
python3 evaluate_pvd4_models.py
```

### 기능
- 저장된 LSTM과 PatchTST 모델 로드
- 검증 데이터셋에서 성능 평가
- 컬럼별 RMSE, MAE, MAPE 계산
- 승자 결정 및 결과 저장

### 출력 파일
- `models/PVD4/comparison_aug50_results.txt`

### 출력 예시
```
======================================================================
 COMPARISON RESULTS (Augmentation = 50)
======================================================================

Model        Val Loss       Val MAE        Best Epoch
----------------------------------------------------
LSTM         0.004114       0.019060       100
PatchTST     0.008623       0.026356       72

======================================================================
 >>> LSTM WINS! (Val Loss 52.3% lower) <<<
======================================================================
```

---

## 2. 예측 결과 시각화

### 실행 명령
```bash
cd /home/goo4168/baco/train
python3 visualize_prediction.py
```

### 기능
- 실제값 vs 예측값 차트 생성
- 타겟 컬럼별 시계열 비교
- PNG 파일로 저장

### 출력 파일
- `models/PVD4/PVD4_prediction_chart.png`

---

## 3. 검증 순서

### 기본 검증 절차
```bash
# 1. 모델 파일 존재 확인
ls -la models/PVD4/

# 2. LSTM vs PatchTST 비교 평가
python3 evaluate_pvd4_models.py

# 3. 결과 확인
cat models/PVD4/comparison_aug50_results.txt

# 4. 시각화 (선택)
python3 visualize_prediction.py
```

### 확장 모델 검증
```bash
# 결과 파일 확인
cat models/PVD4/lstm_aug100_extended_results.txt
```

---

## 4. 평가 지표 설명

| 지표 | 설명 | 좋은 값 |
|------|------|---------|
| **Val Loss** | 검증 손실 (MSE) | 낮을수록 좋음 |
| **Val MAE** | 평균 절대 오차 (정규화) | 낮을수록 좋음 |
| **RMSE** | 제곱근 평균 제곱 오차 | 낮을수록 좋음 |
| **MAE** | 평균 절대 오차 (원본 스케일) | 낮을수록 좋음 |
| **MAPE** | 평균 절대 백분율 오차 | 낮을수록 좋음 (10% 이하 권장) |

---

## 5. 모델별 성능 비교

### 증강 50 기준 (4개 타겟)
| 모델 | Val Loss | 개선율 |
|------|----------|--------|
| LSTM | 0.004114 | 기준 |
| PatchTST | 0.008623 | -52.3% |

**결론**: LSTM이 PatchTST보다 52.3% 더 좋은 성능

### 증강별 LSTM 성능
| 증강 | Val Loss | 개선율 |
|------|----------|--------|
| 없음 | 0.008933 | 기준 |
| 50x | 0.004050 | +54.7% |
| 100x (7타겟) | 0.004110 | +54.0% |

---

## 6. 커스텀 평가

### 새로운 테스트 데이터로 평가
```python
import torch
import joblib
import numpy as np

# 모델 로드
model_path = 'models/PVD4/lstm_aug100_extended_best.pth'
checkpoint = torch.load(model_path)

# 스케일러 로드
scaler_input = joblib.load('models/PVD4/scaler_input_aug100_extended.pkl')
scaler_target = joblib.load('models/PVD4/scaler_target_aug100_extended.pkl')

# 예측
input_scaled = scaler_input.transform(your_input_data)
# ... 모델 추론 ...
output_orig = scaler_target.inverse_transform(output)
```

---

## 7. 트러블슈팅

### 모델 파일 없음
```bash
# 모델 학습 먼저 실행
python3 train_pvd4_extended.py
```

### 스케일러 파일 없음
학습 시 자동 생성됩니다. 학습을 다시 실행하세요.

### CUDA 메모리 부족
```python
# CPU로 평가
device = torch.device('cpu')
checkpoint = torch.load(model_path, map_location=device)
```

---

## 8. 결과 해석

### 좋은 성능 기준
- **MAPE < 10%**: 우수
- **MAPE 10-20%**: 양호
- **MAPE > 20%**: 개선 필요

### Ion.Gauge.i MAPE가 높은 이유
- 값이 매우 작아 (0.0000x) 백분율 오차가 증폭됨
- RMSE/MAE로 평가하는 것이 더 적절

### SBRF5.Reflect MAPE가 높은 이유
- 값의 변동 범위가 작아 상대 오차가 큼
- 절대 오차(MAE: 0.47)는 매우 작음
