# ALD 공정 PatchTST 시계열 예측 — 가이드

> ALD(Atomic Layer Deposition) 공정 트레이스 데이터를 이용해 PatchTST 기반 Transformer 모델로 센서 값을 **10·20·30초 후** 동시 예측합니다.

---

## 데이터 받는 방법

### 서버 정보

| 항목 | 내용 |
|------|------|
| 호스트 | `bigsoft.iptime.org` |
| SSH 포트 | `7784` |
| 계정 | `keti` |
| 데이터 경로 | `/data1/data/standard_trace` |

### 1. GitHub에서 코드 받기

```bash
git clone git@github.com:ChulseoungChae/semi_ondevice.git
cd semi_ondevice/ald
```

HTTPS를 사용하는 경우:

```bash
git clone https://github.com/ChulseoungChae/semi_ondevice.git
cd semi_ondevice/ald
```

### 2. rsync로 학습 데이터 다운로드

로컬 PC에서 아래 명령을 실행합니다. `<로컬_저장_경로>`는 데이터를 받을 디렉터리로 바꿉니다.

```bash
mkdir -p <로컬_저장_경로>/standard_trace

rsync -avz --progress \
  -e "ssh -p 7784" \
  keti@bigsoft.iptime.org:/data1/data/standard_trace/ \
  <로컬_저장_경로>/standard_trace/
```

예시 (현재 디렉터리 기준 상위에 데이터 폴더 생성):

```bash
mkdir -p ../standard_TraceData_80

rsync -avz --progress \
  -e "ssh -p 7784" \
  keti@bigsoft.iptime.org:/data1/data/standard_trace/ \
  ../standard_TraceData_80/
```

> **참고:** 학습·추론 코드는 기본적으로 `../standard_TraceData_80` 경로의 CSV를 읽습니다.  
> rsync로 다른 경로에 받았다면, 해당 경로로 심볼릭 링크를 만들거나 코드/노트북의 데이터 경로를 수정하세요.

```bash
# 심볼릭 링크 예시
ln -s /path/to/standard_trace ../standard_TraceData_80
```

### 데이터 개요

- 파일 형식: CSV (`standard_trace_001.csv` ~ `standard_trace_074.csv`, 총 74개)
- 주요 컬럼: `Step Name`, `Date`, `Time`, MFC 유량·위치, Baratron Gauge 압력(`VG11`~`VG13`), 온도(`TempAct_*`, `TempSet_*`), 밸브 상태 등
- 샘플링: 코드에서 `iloc[1::2]`로 **1행 걸러 1행** 사용 (2초 간격 → 1초 간격)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [코드 개요](#3-코드-개요)
4. [환경 설정](#4-환경-설정)
5. [학습 실행 방법](#5-학습-실행-방법)
6. [추론 실행 방법](#6-추론-실행-방법)
7. [산출물](#7-산출물)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | ALD 공정 센서 시계열의 미래 값 예측 (이상 감지·공정 모니터링 기반) |
| 모델 | PatchTST 변형 Transformer (Keras) |
| 입력 | 과거 192초(192 timestep) 다변량 시계열 |
| 출력 | 10·20·30초 후 타깃 센서 값 3개 동시 예측 |
| 프레임워크 | TensorFlow 2 / Keras, MirroredStrategy 멀티 GPU, mixed_float16 |

### 예측 대상 센서 (학습 코드 기준)

`VG11`, `MFCMon_P.POS`, `MFCMon_L.POS`, `VG12`, `VG13`, `MFCMon_N2-1`~`N2-4`, `APCValveMon`

---

## 2. 디렉터리 구조

```
semi_ondevice/ald/
├── patchTST_v2.py              # 학습 코드
├── validation_patchTST_v2.ipynb # 추론·검증 노트북
└── readme.md                   # 본 문서

# 학습 실행 후 생성되는 디렉터리 (patchTST_v2.py 기준)
./scaler_v5/                    # MinMaxScaler 저장
./patchtst_model_v5/            # 학습된 .keras 모델
log.txt                         # 학습 로그

# 추론 노트북 기준 (별도 준비 필요)
./model/                        # 추론용 모델
./model/scaler/                 # 추론용 스케일러
./newdata3/                     # 추론 결과 CSV 저장
```

서버의 추론 작업 디렉터리: `/data1/standard_train/`  
(동일한 `validation_patchTST_v2.ipynb` 및 `model/`, `model2/` 디렉터리 포함)

---

## 3. 코드 개요

### 3.1 학습 코드 — `patchTST_v2.py`

ALD 공정 CSV를 읽어 **타깃 센서별 PatchTST 모델**을 학습하고 저장합니다.

**처리 흐름:**

1. `../standard_TraceData_80` 하위 CSV 수집 → 파일명 정렬 후 **마지막 5개는 테스트용**으로 분리
2. `global_min` / `global_max` 기반 MinMaxScaler 생성 → `./scaler_v5/` 저장
3. `predict_columns` 각 센서에 대해 반복:
   - `Step Name` → `Step ID` 정수 변환 (`step_reverse_dict`)
   - 1/2 다운샘플링, 결측 제거, 정규화
   - 슬라이딩 윈도우(`window_size=192`)로 시퀀스 생성
   - **20개 CSV마다** 누적 데이터로 `model.fit` (증분 학습)
   - EarlyStopping + ReduceLROnPlateau 적용
4. 모델을 `./patchtst_model_v5/mae_192_patchtst_{센서}_main.keras`로 저장

**모델 구조:**

```
입력 (batch, 192, num_features)
  → PatchEmbedding (patch_len=16, d_model=64)
  → PositionalEncoding
  → Transformer Block × 2 (MultiHeadAttention + FFN)
  → GlobalAveragePooling1D
  → Dense(64, GELU) → Dense(3)   # 10·20·30초 후 예측
```

**주요 하이퍼파라미터:**

| 파라미터 | 값 |
|----------|-----|
| `window_size` | 192 |
| `predict_steps` | [10, 20, 30] |
| `patch_len` | 16 |
| `d_model` | 64 |
| `num_heads` | 4 |
| `ff_dim` | 128 |
| `num_layers` | 2 |
| `batch_size` | 256 |
| `epochs_per_run` | 200 (EarlyStopping으로 조기 종료) |
| `learning_rate` | 5e-4 (온도 계열: 1e-3) |

**특수 처리:**

- `VG11`: 특정 압력 구간에 가중치 100을 주는 weighted MAE 손실 사용
- `Temp*` 계열: `TempSet_*`, `Power_HT.*` 추가 입력 + 전용 `scaler_X` 사용

---

### 3.2 추론 코드 — `validation_patchTST_v2.ipynb`

학습된 모델을 로드해 테스트 CSV에 대해 예측하고, MAE/MSE를 계산하며 결과를 CSV로 저장합니다.

**처리 흐름:**

1. 라이브러리·하이퍼파라미터·`PatchEmbedding` / `PositionalEncoding` 커스텀 레이어 정의
2. 테스트 CSV 로드 (`test_csv_list`, 기본: 전체 CSV 중 인덱스 60~62)
3. 센서(`predict_columns`)별로:
   - `model/{window_size}_patchtst_{센서}.keras` 로드
   - `model/scaler/`에서 스케일러 로드
   - 전처리(Step ID 변환, 1/2 샘플링, 정규화, 시퀀스 생성) 후 `predict`
   - N2·POS·VG12·VG13 등 일부 센서는 **메인 공정 Step** 구간에서 `*_main.keras` 보정 모델로 예측값 덮어쓰기
4. 실측·10s·20s·30s 예측 결과를 `./newdata3/{파일명}_actual.csv` 등 4개 파일로 저장
5. matplotlib으로 Actual vs Predicted 그래프 출력

**보정 모델 적용 대상 (`check_columns`):**

- `MFCMon_N2-1` ~ `N2-4`
- `MFCMon_L.POS`, `MFCMon_P.POS`, `VG12`, `VG13`

메인 공정 Step ID: 111, 128, 119, 117, 152, 113, 115, 116

---

## 4. 환경 설정

### Python 패키지

```bash
pip install tensorflow scikit-learn pandas numpy joblib matplotlib jupyter
```

GPU 학습 시 CUDA 호환 TensorFlow 버전을 설치하세요.

### GPU 설정 (학습 코드)

`patchTST_v2.py` 상단에서 사용할 GPU를 지정합니다.

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 사용할 GPU ID
```

GPU가 없는 환경에서는 `"CUDA_VISIBLE_DEVICES" = ""`로 CPU만 사용할 수 있습니다.

---

## 5. 학습 실행 방법

### 사전 준비

1. [데이터 받는 방법](#데이터-받는-방법)대로 CSV를 `../standard_TraceData_80/`에 배치
2. Python 환경 및 TensorFlow 설치

### 실행

```bash
cd semi_ondevice/ald
python patchTST_v2.py
```

학습은 `predict_columns`에 정의된 센서마다 순차적으로 진행되며, 시간이 오래 걸릴 수 있습니다.

### 데이터 경로 변경

다른 경로에 데이터가 있다면 `patchTST_v2.py` 36번째 줄을 수정합니다.

```python
data_path = "../standard_TraceData_80"  # ← 실제 데이터 경로로 변경
```

### 학습 결과 확인

```bash
ls ./scaler_v5/
ls ./patchtst_model_v5/
cat log.txt
```

---

## 6. 추론 실행 방법

### 사전 준비

1. 테스트용 CSV (`../standard_TraceData_80/`)
2. 학습된 모델·스케일러를 추론 노트북이 기대하는 경로에 배치

추론 노트북은 아래 구조의 모델을 사용합니다.

```
model/
├── 192_patchtst_{센서}.keras
├── mae_192_patchtst_{센서}_main.keras   # 보정 모델 (일부 센서)
└── scaler/
    ├── scaler_X.pkl
    ├── scaler_X_main.pkl
    ├── scaler_X_{온도센서}.pkl
    ├── scaler_y_{센서}.pkl
    └── scaler_y_{센서}_main.pkl
```

> 학습 코드(`patchTST_v2.py`)는 `./patchtst_model_v5/`, `./scaler_v5/`에 저장합니다.  
> 추론 전에 파일명·경로를 `model/` 구조에 맞게 복사·이름 변경하거나, 노트북의 경로를 수정해야 합니다.

서버(`/data1/standard_train/`)에는 이미 학습된 `model/`, `model2/` 디렉터리가 있습니다.  
로컬에서 추론하려면 모델 파일도 rsync로 함께 받을 수 있습니다.

```bash
rsync -avz --progress \
  -e "ssh -p 7784" \
  keti@bigsoft.iptime.org:/data1/standard_train/model/ \
  ./model/

rsync -avz --progress \
  -e "ssh -p 7784" \
  keti@bigsoft.iptime.org:/data1/standard_train/validation_patchTST_v2.ipynb \
  ./
```

### Jupyter Notebook 실행

```bash
cd semi_ondevice/ald   # 또는 standard_train
jupyter notebook validation_patchTST_v2.ipynb
```

또는 JupyterLab:

```bash
jupyter lab validation_patchTST_v2.ipynb
```

### 노트북 실행 순서

| 셀 | 내용 |
|----|------|
| 0 | 라이브러리 import |
| 1 | `window_size`, `predict_steps`, `predict_columns` 등 파라미터 설정 |
| 2~3 | 유틸 함수·`selected_cols` 정의 |
| 5 | 테스트 CSV 목록 로드 |
| 6~7 | 커스텀 레이어·손실 함수 정의 |
| 8 | `check_columns`, `is_main_proc` (보정 모델 판별) |
| 11 | **메인 추론 루프** — 예측·평가·CSV 저장·그래프 출력 |

셀 10은 `model2/` 파일 이름 변환용 유틸리티이며, 셀 13~16은 별도 실험용입니다.

### 테스트 파일 범위 변경

노트북 셀 5에서 `test_csv_list` 슬라이스를 수정합니다.

```python
csv_list = find_csv_files('../standard_TraceData_80')
csv_list.sort()
test_csv_list = csv_list[60:63]  # 원하는 인덱스 범위로 변경
```

### 추론 결과

`./newdata3/` 디렉터리에 파일별로 4개 CSV가 생성됩니다.

| 파일 | 내용 |
|------|------|
| `{파일명}_actual.csv` | 실측값 (현재 시점) |
| `{파일명}_pred_10s.csv` | 10초 후 예측 |
| `{파일명}_pred_20s.csv` | 20초 후 예측 |
| `{파일명}_pred_30s.csv` | 30초 후 예측 |

---

## 7. 산출물

### 학습 (`patchTST_v2.py`)

| 경로 | 설명 |
|------|------|
| `./scaler_v5/scaler_X_main.pkl` | 입력 특성 공통 MinMaxScaler |
| `./scaler_v5/scaler_y_{센서}.pkl` | 타깃별 MinMaxScaler |
| `./scaler_v5/scaler_X_{센서}.pkl` | 온도 타깃용 확장 입력 스케일러 |
| `./patchtst_model_v5/mae_192_patchtst_{센서}_main.keras` | 학습된 모델 |
| `log.txt` | 학습 진행 로그 |

### 추론 (`validation_patchTST_v2.ipynb`)

| 경로 | 설명 |
|------|------|
| `./newdata3/{파일명}_actual.csv` | 실측 결과 |
| `./newdata3/{파일명}_pred_10s.csv` | 10초 후 예측 |
| `./newdata3/{파일명}_pred_20s.csv` | 20초 후 예측 |
| `./newdata3/{파일명}_pred_30s.csv` | 30초 후 예측 |

---

## 참고

- 학습 코드는 테스트 CSV 5개를 분리만 하고, 평가는 추론 노트북에서 수행합니다.
- 추론 노트북의 `predict_columns` 목록은 학습 코드와 다를 수 있습니다. 사용할 센서에 맞게 양쪽을 일치시키세요.
- `protobuf` 버전 충돌 시 `'MessageFactory' object has no attribute 'GetPrototype'` 오류가 날 수 있습니다. `pip install protobuf==3.20.3` 등으로 버전을 맞춰 보세요.
