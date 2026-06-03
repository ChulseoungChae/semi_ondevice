# semi_ondevice — 반도체 공정 On-Device AI

ALD·PVD 장비 트레이스 데이터를 이용한 **시계열 예측·이상 감지** 코드 모음입니다.

| 프로젝트 | 장비 | 목적 | 상세 문서 |
|----------|------|------|-----------|
| **ALD** | ALD 챔버 | PatchTST 기반 센서 **10·20·30초 후** 예측 | [ald/readme.md](ald/readme.md) |
| **PVD** | PVD4 챔버 | 실시간 이상 감지 웹 플랫폼 + LSTM/CNN/Aging 모델 | [pvd/readme.md](pvd/readme.md) |

---

## 저장소 구조

```
semi_ondevice/
├── readme.md                          # 본 문서
├── ald/
│   ├── patchTST_v2.py                 # PatchTST 학습
│   ├── validation_patchTST_v2.ipynb   # 추론·검증
│   └── readme.md
└── pvd/
    ├── pvd_monitoring/                # FastAPI 웹 플랫폼 (추론·모니터링)
    ├── train/
    │   └── train_7feat_aging.py       # Aging LSTM 학습
    ├── lstm_project/                  # LSTM/CNN 모델·스케일러·bounds
    └── readme.md
```

---

## 코드 받기

```bash
git clone git@github.com:ChulseoungChae/semi_ondevice.git
cd semi_ondevice
```

HTTPS:

```bash
git clone https://github.com/ChulseoungChae/semi_ondevice.git
cd semi_ondevice
```

---

## 데이터 받는 방법

ALD와 PVD는 서버·데이터 경로가 다릅니다.

### ALD 데이터

| 항목 | 내용 |
|------|------|
| 서버 | `ssh keti@bigsoft.iptime.org -p 7784` |
| 데이터 경로 | `/data1/data/standard_trace` |
| 파일 | `standard_trace_001.csv` ~ `standard_trace_074.csv` (74개) |

```bash
mkdir -p ../standard_TraceData_80

rsync -avz --progress \
  -e "ssh -p 7784" \
  keti@bigsoft.iptime.org:/data1/data/standard_trace/ \
  ../standard_TraceData_80/
```

> 학습·추론 코드는 기본 경로 `../standard_TraceData_80`을 사용합니다. 다른 위치에 받았다면 심볼릭 링크를 만들거나 코드/노트북의 경로를 수정하세요.

학습된 ALD 모델(추론용):

```bash
rsync -avz --progress \
  -e "ssh -p 7784" \
  keti@bigsoft.iptime.org:/data1/standard_train/model/ \
  ald/model/
```

### PVD 데이터

| 항목 | 내용 |
|------|------|
| 서버 | `ssh goo4168@bigsoft.time.org -p 7783` |
| 레시피 실험 데이터 | `/home/goo4168/baco/PDS_Data_Log/` (45개 CSV) |
| 실제 운용 데이터 | `/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/` (243개 CSV) |

```bash
# 레시피 실험 데이터
scp -P 7783 -r goo4168@bigsoft.time.org:/home/goo4168/baco/PDS_Data_Log/ ./pvd/data/PDS_Data_Log/

# 실제 운용 데이터 (특정 월 예시)
scp -P 7783 -r goo4168@bigsoft.time.org:/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/2026_01/ ./pvd/data/minio/2026_01/
```

학습된 PVD 모델:

```bash
scp -P 7783 -r goo4168@bigsoft.time.org:/home/goo4168/baco/lstm_project/models/ ./pvd/lstm_project/models/
scp -P 7783 -r goo4168@bigsoft.time.org:/home/goo4168/baco/lstm_project/results/ ./pvd/lstm_project/results/
scp -P 7783 -r goo4168@bigsoft.time.org:/home/goo4168/baco/train/models/PVD4/ ./pvd/train/models/PVD4/
```

---

# ALD — PatchTST 시계열 예측

## 개요

ALD 공정 CSV 트레이스를 읽어 **PatchTST 변형 Transformer**로 센서 값을 **10·20·30초 후** 동시 예측합니다.

| 항목 | 내용 |
|------|------|
| 학습 | `ald/patchTST_v2.py` |
| 추론 | `ald/validation_patchTST_v2.ipynb` |
| 프레임워크 | TensorFlow 2 / Keras, MirroredStrategy, mixed_float16 |
| 입력 | 과거 192 timestep 다변량 시계열 |
| 출력 | 10·20·30초 후 타깃 센서 값 3개 |

### 예측 대상 센서 (학습 코드 기준)

`VG11`, `MFCMon_P.POS`, `MFCMon_L.POS`, `VG12`, `VG13`, `MFCMon_N2-1`~`N2-4`, `APCValveMon`

### 코드 흐름

**학습 (`patchTST_v2.py`)**

1. CSV 수집 → 마지막 5개는 테스트용 분리
2. MinMaxScaler 생성 → `./scaler_v5/` 저장
3. 센서별 PatchTST 모델 학습 (20개 CSV마다 증분 `fit`)
4. `./patchtst_model_v5/mae_192_patchtst_{센서}_main.keras` 저장

**추론 (`validation_patchTST_v2.ipynb`)**

1. `model/`에서 학습된 `.keras` 모델·스케일러 로드
2. 테스트 CSV 전처리 → 예측 → MAE/MSE 계산
3. 일부 센서(N2, POS, VG12/13)는 메인 공정 Step에서 `*_main.keras` 보정 모델 적용
4. `./newdata3/`에 actual / pred_10s / pred_20s / pred_30s CSV 저장

### 환경 설정

```bash
pip install tensorflow scikit-learn pandas numpy joblib matplotlib jupyter
```

### 실행 방법

**학습**

```bash
cd ald
python patchTST_v2.py
```

**추론**

```bash
cd ald
jupyter notebook validation_patchTST_v2.ipynb
# 또는
jupyter lab validation_patchTST_v2.ipynb
```

노트북 셀 11이 메인 추론 루프입니다. 학습 산출물(`patchtst_model_v5/`, `scaler_v5/`)은 추론 전 `model/` 구조에 맞게 복사·이름 변경이 필요합니다.

---

# PVD — 공정 이상 감지 플랫폼

## 개요

PVD4 장비 센서 데이터를 실시간 모니터링하여 **공정 이상·장비 노후화**를 조기 감지하는 AI 기반 웹 플랫폼입니다.

| 감지 유형 | 방법 | 기준 |
|-----------|------|------|
| 마그넷 오염 | 규칙 기반 | OES 감소 + DC Power 유지/증가 |
| 가스 유량 이상 | 규칙 기반 | Ar.MFC.i 안정 구간 대비 ±1.5% 초과 |
| Ar 유량 진동 | 규칙 기반 | ±1% 초과 진동 4초 이상 지속 |
| PWPDS 예측 이상 | CNN | 예측값 대비 5%/10% 이상 오차 |
| 장비 노후 | LSTM (7feat) | 5초 후 예측 대비 10% 이상 오차 |

### 주요 구성

| 경로 | 역할 |
|------|------|
| `pvd/pvd_monitoring/main.py` | FastAPI 앱, REST API, WebSocket |
| `pvd/pvd_monitoring/abnormal_monitor.py` | CSV 실시간 감시 + 이상 감지 조율 |
| `pvd/pvd_monitoring/cnn_inference.py` | CNN PWPDS 예측 이상 감지 |
| `pvd/pvd_monitoring/lstm_inference.py` | LSTM PWPDS 예측 |
| `pvd/pvd_monitoring/aging_inference.py` | Aging 7센서 LSTM 추론 |
| `pvd/pvd_monitoring/models/anomaly_detector.py` | 규칙 기반 이상 감지 |
| `pvd/pvd_monitoring/generate_data.py` | 테스트 데이터 재생기 |
| `pvd/train/train_7feat_aging.py` | Aging LSTM 학습 |

### 모델 요약

| 모델 | 목적 | 입력 | 출력 |
|------|------|------|------|
| LSTM (`model_B_aug100x`) | 5초 후 PWPDS 예측 | 10초 × 5 feature | PWPDS.Data |
| CNN (`v3_CNN_alldata_aug50x`) | PWPDS 예측 + bounds 이상 | 10초 × 8 feature | PWPDS.Data |
| Aging LSTM (`lstm_7feat`) | 7센서 5초 후 예측 | 10초 × 7 feature | 7개 센서 |

### 환경 설정

```bash
cd pvd/pvd_monitoring
./run.sh install
# PyTorch 모델 추론용 (학습·추론 엔진)
pip install torch pandas numpy scikit-learn joblib
```

### 실행 방법

**웹 서버 (추론·모니터링)**

```bash
cd pvd/pvd_monitoring
./run.sh install    # 최초 1회
./run.sh server     # http://0.0.0.0:9300
```

운영 서버 접속: `http://bigsoft.time.org:9300`

**데모 모드 (테스트 데이터 생성 + 웹 서버)**

```bash
cd pvd/pvd_monitoring
./run.sh demo
```

**테스트 데이터 생성기만**

```bash
./run.sh generate                    # 단일 공정 (정상)
./run.sh generate --continuous       # 연속 생성
./run.sh generate --inject-anomaly   # 이상 데이터 주입
```

**CLI 모니터링**

```bash
./run.sh monitor
```

**Aging LSTM 학습**

```bash
cd pvd/train
python3 train_7feat_aging.py
```

> 학습 스크립트의 데이터 경로(`MINIO_GLOB`, `PDS_GLOB`, `SAVE_DIR`)는 서버 경로 기준입니다. 로컬에서 학습하려면 경로를 수정하세요.

백그라운드 학습:

```bash
nohup python3 train_7feat_aging.py > /tmp/train_7feat.log 2>&1 &
tail -f /tmp/train_7feat.log
```

### 데이터 흐름 (웹 플랫폼)

```
CSV 생성 (data/)
    ↓
FileWatcher (abnormal_monitor.py)
    ↓
규칙 기반 이상 감지 + CNN 추론 + Aging LSTM 추론
    ↓
WebSocket → 브라우저 차트 갱신
```

---

## 빠른 참조

| 작업 | 명령 |
|------|------|
| ALD 학습 | `cd ald && python patchTST_v2.py` |
| ALD 추론 | `cd ald && jupyter notebook validation_patchTST_v2.ipynb` |
| PVD 웹 서버 | `cd pvd/pvd_monitoring && ./run.sh server` |
| PVD 데모 | `cd pvd/pvd_monitoring && ./run.sh demo` |
| PVD Aging 학습 | `cd pvd/train && python3 train_7feat_aging.py` |

---

## 상세 문서

- ALD 상세: [ald/readme.md](ald/readme.md)
- PVD 상세: [pvd/readme.md](pvd/readme.md)
- PVD 웹 플랫폼: [pvd/pvd_monitoring/README.md](pvd/pvd_monitoring/README.md)
