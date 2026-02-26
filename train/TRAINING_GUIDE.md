# PVD4 모델 학습 가이드

## 개요
PVD4 공정 데이터에 대한 LSTM 예측 모델 학습 가이드입니다.

---

## 파일 구조

```
train/
├── train_pvd4_extended.py     # 확장 모델 학습 (7개 타겟, 증강 100)
├── train_pvd4_compare.py      # LSTM vs PatchTST 비교 학습
├── evaluate_pvd4_models.py    # 모델 평가/검증
├── visualize_prediction.py    # 예측 결과 시각화
├── inference.py               # 추론 모듈
├── pvd_predictor.py           # 예측기 클래스
└── models/
    └── PVD4/
        ├── lstm_aug100_extended_best.pth  # 최종 모델 (권장)
        ├── scaler_input_aug100_extended.pkl
        └── scaler_target_aug100_extended.pkl
```

---

## 1. 확장 모델 학습 (권장)

### 실행 명령
```bash
cd /home/goo4168/baco/train
python3 train_pvd4_extended.py
```

### 설정
| 항목 | 값 |
|------|-----|
| 타겟 컬럼 | 7개 (Ar.MFC.i, Ion.Gauge.i, Baratron.Gauge.i, OES.Data6, PLA5.Match.DCBias, SBRF5.Forward, SBRF5.Reflect) |
| 증강 배수 | 100x |
| 입력 윈도우 | 10 스텝 |
| 예측 호라이즌 | 5 스텝 |
| 에폭 | 100 (Early Stopping 15) |
| 배치 크기 | 64 |
| 학습률 | 0.0005 |

### 출력 파일
- `models/PVD4/lstm_aug100_extended_best.pth` - 학습된 모델
- `models/PVD4/scaler_input_aug100_extended.pkl` - 입력 스케일러
- `models/PVD4/scaler_target_aug100_extended.pkl` - 타겟 스케일러
- `models/PVD4/lstm_aug100_extended_results.txt` - 학습 결과

### 예상 소요 시간
- RTX 4090 기준: 약 3시간

---

## 2. LSTM vs PatchTST 비교 학습

### 실행 명령
```bash
cd /home/goo4168/baco/train
python3 train_pvd4_compare.py
```

### 설정
| 항목 | 값 |
|------|-----|
| 타겟 컬럼 | 4개 (Ar.MFC.i, Ion.Gauge.i, Baratron.Gauge.i, OES.Data6) |
| 증강 배수 | 50x |
| 모델 | LSTM, PatchTST |

### 출력 파일
- `models/PVD4/lstm_aug50_best.pth`
- `models/PVD4/patchtst_aug50_best.pth`
- `models/PVD4/comparison_aug50_results.txt`

---

## 3. 데이터 경로 설정

학습 데이터는 다음 경로에 위치해야 합니다:
```
/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/
```

CSV 파일에 필수 컬럼:
- Ar.MFC.i
- Ion.Gauge.i
- Baratron.Gauge.i
- OES.Data6
- PLA5.Match.DCBias (확장 모델)
- SBRF5.Forward (확장 모델)
- SBRF5.Reflect (확장 모델)

---

## 4. 학습 순서

### 처음 학습하는 경우
```bash
# 1. 필요 패키지 설치
pip install torch numpy pandas scikit-learn joblib

# 2. 확장 모델 학습 (권장)
python3 train_pvd4_extended.py

# 3. 결과 확인
cat models/PVD4/lstm_aug100_extended_results.txt
```

### 모델 비교가 필요한 경우
```bash
# LSTM vs PatchTST 비교
python3 train_pvd4_compare.py

# 결과 확인
cat models/PVD4/comparison_aug50_results.txt
```

---

## 5. GPU 사용

CUDA가 설치되어 있으면 자동으로 GPU를 사용합니다.

```bash
# GPU 확인
nvidia-smi

# CUDA 버전 확인
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 6. 학습 결과 예시

### 확장 모델 (증강 100, 7개 타겟)
| 컬럼 | RMSE | MAE | MAPE |
|------|------|-----|------|
| Ar.MFC.i | 6.70 | 1.57 | 9.55% |
| Ion.Gauge.i | 0.00030 | 0.00009 | - |
| Baratron.Gauge.i | 0.40 | 0.23 | 14.08% |
| OES.Data6 | 0.82 | 0.28 | 25.46% |
| PLA5.Match.DCBias | 37.23 | 12.36 | 3.17% |
| SBRF5.Forward | 24.92 | 5.24 | 0.84% |
| SBRF5.Reflect | 2.95 | 0.47 | 70.95% |

---

## 7. 트러블슈팅

### 메모리 부족
```python
# train_pvd4_extended.py에서 배치 크기 줄이기
BATCH_SIZE = 32  # 64 -> 32
```

### 학습이 너무 느림
```python
# 증강 배수 줄이기
AUGMENTATION_FACTOR = 50  # 100 -> 50
```

### 타겟 컬럼 누락 에러
- CSV 파일에 필수 컬럼이 모두 있는지 확인
- 153개 파일이 스킵된다면 해당 파일들에 새로운 타겟 컬럼이 없는 것
