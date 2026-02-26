# PVD4 공정 이상감지 플랫폼 — 종합 가이드

> 작성일: 2026-02-27
> 대상 장비: PVD4 (물리 기상 증착 챔버)

---

## 서버 접속 정보

| 항목 | 내용 |
|------|------|
| 호스트 | `bigsoft.time.org` |
| SSH 포트 | `7783` |
| 계정 | `goo4168` |
| 웹 포트 | `9300` |

**SSH 접속:**
```bash
ssh goo4168@bigsoft.time.org -p 7783
```

**웹 대시보드 접속:**
```
http://bigsoft.time.org:9300
```

**파일 복사 (로컬 → 서버):**
```bash
scp -P 7783 <로컬파일> goo4168@bigsoft.time.org:<원격경로>
```

**파일 복사 (서버 → 로컬):**
```bash
scp -P 7783 goo4168@bigsoft.time.org:<원격경로> <로컬경로>
```

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [학습 데이터](#3-학습-데이터)
4. [모델 정보](#4-모델-정보)
5. [학습 방법](#5-학습-방법)
6. [웹 플랫폼 기능](#6-웹-플랫폼-기능)
7. [실행 방법](#7-실행-방법)
8. [주요 파일 참조](#8-주요-파일-참조)

---

## 1. 프로젝트 개요

PVD4 장비의 센서 데이터를 실시간으로 모니터링하여 공정 이상 및 장비 노후화를 조기에 감지하는 AI 기반 웹 플랫폼.

### 감지 대상

| 감지 유형 | 방법 | 기준 |
|-----------|------|------|
| 마그넷 오염 | 규칙 기반 | OES 감소 + DC Power 유지/증가 |
| 가스 유량 이상 | 규칙 기반 | Ar.MFC.i 안정 구간 대비 ±1.5% 초과 |
| Ar 유량 진동 (AR_FLUCTUATION) | 규칙 기반 | ±1% 초과 진동이 4초 이상 지속 |
| PWPDS 예측 이상 | CNN 모델 | 예측값 대비 5%/10% 이상 오차 |
| 장비 노후 감지 | LSTM 모델 (7feat) | 센서 5초 후 예측값 대비 10% 이상 오차 |

---

## 2. 디렉토리 구조

**서버:** `goo4168@bigsoft.time.org -p 7783`
**기본 경로:** `/home/goo4168/baco/`

```
/home/goo4168/baco/
├── pvd_monitoring/          # 웹 플랫폼 (FastAPI)
│   ├── main.py              # FastAPI 앱 진입점, REST API, WebSocket
│   ├── abnormal_monitor.py  # 실시간 CSV 감시 + 이상감지 조율
│   ├── lstm_inference.py    # LSTM 추론 엔진 (PWPDS.Data 예측)
│   ├── cnn_inference.py     # CNN 추론 엔진 (PWPDS.Data 예측)
│   ├── aging_inference.py   # Aging LSTM 추론 엔진 (7개 센서 예측)
│   ├── generate_data.py     # 테스트 데이터 생성기
│   ├── models/
│   │   └── anomaly_detector.py  # 규칙 기반 이상감지 로직
│   ├── templates/
│   │   └── index.html       # 웹 대시보드 (Chart.js)
│   ├── data/                # 감시 대상 CSV 파일 저장 위치
│   ├── logs/                # 이상 로그 저장
│   ├── exports/             # CSV 내보내기
│   └── run.sh               # 실행 스크립트
│
├── lstm_project/            # 기존 PWPDS 예측 모델 (웹 연동)
│   ├── models/
│   │   ├── model_B_aug100x_best.pt          # LSTM 모델 (웹 사용)
│   │   ├── model_B_aug100x_scaler_x.pkl     # LSTM 입력 스케일러
│   │   ├── model_B_aug100x_scaler_y.pkl     # LSTM 출력 스케일러
│   │   ├── v3_CNN_alldata_aug50x_best.pt    # CNN 모델 (웹 사용)
│   │   ├── v3_CNN_alldata_aug50x_scaler_x.pkl
│   │   └── v3_CNN_alldata_aug50x_scaler_y.pkl
│   └── results/
│       ├── bounds_lookup.json   # DC/RF 레시피별 PWPDS 상하한 (웹 사용)
│       └── column_ranges.json   # 차트 Y축 범위값 (웹 사용)
│
├── train/                   # Aging 모델 학습
│   ├── train_7feat_aging.py         # Aging LSTM 학습 스크립트 (현행)
│   └── models/PVD4/
│       ├── lstm_7feat_best.pth              # Aging 모델 (웹 사용)
│       ├── scaler_input_lstm_7feat.pkl      # 입력 스케일러
│       └── scaler_target_lstm_7feat.pkl     # 출력 스케일러
│
├── PDS_Data_Log/            # 레시피별 공정 데이터 (45개 CSV, 1초 간격)
├── minio_csv/               # 실제 장비 운용 데이터
│   └── Baco_origin_3/DataLog/PVD4/
│       └── YYYY_MM/         # 월별 디렉토리 (2025_04 ~ 2026_01)
│           └── PVD4_*.csv   # 총 243개
└── backup/                  # 미사용 구버전 모델/코드 보관
```

---

## 3. 학습 데이터

### 3-1. PDS_Data_Log (레시피 실험 데이터)

| 항목 | 내용 |
|------|------|
| 서버 경로 | `goo4168@bigsoft.time.org -p 7783` → `/home/goo4168/baco/PDS_Data_Log/` |
| 파일 수 | 45개 CSV + 45개 TXT (헤더/메타) |
| 샘플링 | 1초 간격 |
| 공정당 행 수 | 약 68~73행 (공정 활성 구간) |
| 레시피 구성 | DC(1000/2000/3000) × RF(0/300/400/500) × 3회 반복 + DCign × RF(300/400/500) × 3회 |
| 주요 컬럼 | Timer, EN4.Power, SBRF5.SetPower, PLA5.Match.DCBias, Ar.MFC.i, Ion.Gauge.i, Baratron.Gauge.i, PWPDS.Data, OES.Data6 등 32개 |

**파일명 패턴:**
```
DC1000(1).csv          → DC 1000W, RF 없음, 1번째 공정
DC1000RF300(2).csv     → DC 1000W, RF 300W, 2번째 공정
DCignRF400(1).csv      → DC 없음, RF 점화 400W, 1번째 공정
```

**서버에서 직접 접근:**
```bash
ssh goo4168@bigsoft.time.org -p 7783
ls /home/goo4168/baco/PDS_Data_Log/
```

**로컬로 다운로드:**
```bash
scp -P 7783 -r goo4168@bigsoft.time.org:/home/goo4168/baco/PDS_Data_Log/ ./
```

### 3-2. minio_csv (실제 운용 데이터)

| 항목 | 내용 |
|------|------|
| 서버 경로 | `goo4168@bigsoft.time.org -p 7783` → `/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/` |
| 월별 디렉토리 | 2025_04, 2025_09, 2025_10, 2025_12, 2026_01 |
| 파일 수 | 243개 CSV (PVD4_*.csv) |
| 샘플링 | 1초 간격, 파일당 수백~수천 행 |
| 컬럼 구조 | PDS_Data_Log와 동일 (단, PWPDS.Data2~8 없을 수 있음) |

**파일명 패턴:**
```
PVD4_NEW_20251013_140520.csv    → 신규 레시피 공정 로그
PVD4_251015_20251015_154346.csv → 날짜 코드 포함 로그
```

**서버에서 직접 접근:**
```bash
ssh goo4168@bigsoft.time.org -p 7783
ls /home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/
```

**특정 월 데이터 다운로드:**
```bash
scp -P 7783 -r goo4168@bigsoft.time.org:/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/2026_01/ ./
```

### 3-3. 데이터 증강

현행 학습 스크립트(`train_7feat_aging.py`)는 **복사 후 이어붙이기 방식**으로 50배 증강:

```
원본 데이터 → 동일 데이터 50회 순차 연결 → 증강 완료
(노이즈 없음, 순수 반복)
```

---

## 4. 모델 정보

### 4-1. LSTM PWPDS 예측 모델 (model_B_aug100x)

| 항목 | 내용 |
|------|------|
| 목적 | 5초 후 PWPDS.Data 예측 → 예측 이상 감지 |
| 아키텍처 | Bidirectional LSTM + Attention |
| 입력 | 10초 윈도우 × 5개 feature |
| 입력 컬럼 | Ar.MFC.i, EN4.Power, SBRF5.SetPower, PLA5.Match.DCBias, PWPDS.Data |
| 출력 | PWPDS.Data (5초 후) |
| 학습 데이터 | PDS_Data_Log 45개 파일 |
| 증강 | 100배 |
| 이상 판정 | 실제값 대비 예측 오차 10% 이상 |
| 서버 모델 경로 | `/home/goo4168/baco/lstm_project/models/model_B_aug100x_best.pt` |
| 서버 스케일러 | `/home/goo4168/baco/lstm_project/models/model_B_aug100x_scaler_x/y.pkl` |

### 4-2. CNN PWPDS 예측 모델 (v3_CNN_alldata_aug50x)

| 항목 | 내용 |
|------|------|
| 목적 | 현재 시점 PWPDS.Data 예측 → bounds 기반 이상 감지 |
| 아키텍처 | 1D CNN |
| 입력 | 10초 윈도우 × 8개 feature |
| 입력 컬럼 | SBRF5.SetPower, EN4.Power, Ar.MFC.i, PLA5.Match.Tune.Posi, PLA5.Match.DCBias, ULVAC.Stage1.Temp1, EN4.Volt, Ion.Gauge.i |
| 출력 | PWPDS.Data (현재 시점, predict_ahead=0) |
| 학습 데이터 | PDS_Data_Log + minio_csv 전체 |
| 증강 | 50배 |
| 이상 판정 | DC/RF 레시피별 bounds 대비 5%/10% 초과 |
| 서버 모델 경로 | `/home/goo4168/baco/lstm_project/models/v3_CNN_alldata_aug50x_best.pt` |
| 서버 스케일러 | `/home/goo4168/baco/lstm_project/models/v3_CNN_alldata_aug50x_scaler_x/y.pkl` |
| 서버 경계값 | `/home/goo4168/baco/lstm_project/results/bounds_lookup.json` |
| 서버 Y축 범위 | `/home/goo4168/baco/lstm_project/results/column_ranges.json` |

### 4-3. Aging LSTM 모델 (lstm_7feat)

| 항목 | 내용 |
|------|------|
| 목적 | 7개 핵심 센서 5초 후 예측 → 장비 노후/열화 감지 |
| 아키텍처 | Bidirectional LSTM + Attention + FC |
| 입력 | 10초 윈도우 × 7개 feature |
| 출력 | 7개 센서값 (5초 후) |
| 입력/출력 컬럼 | Ar.MFC.i, Baratron.Gauge.i, PLA5.Match.DCBias, EN4.Power, SBRF5.SetPower, PWPDS.Data, Ion.Gauge.i |
| 학습 데이터 | minio PVD4 243개 중 유효 131개 + PDS_Data_Log 45개 |
| 증강 | 50배 (복사 이어붙이기) |
| 시퀀스 수 | 2,230,086개 (train 90% / val 10%) |
| 파라미터 수 | 610,184개 |
| 학습 결과 | best val_loss=0.006570 (100 에폭) |
| 이상 판정 | 컬럼별 실제값 대비 예측 오차 10% 이상 |
| 서버 모델 경로 | `/home/goo4168/baco/train/models/PVD4/lstm_7feat_best.pth` |
| 서버 스케일러 | `/home/goo4168/baco/train/models/PVD4/scaler_input/target_lstm_7feat.pkl` |

**Aging 모델 컬럼별 성능:**

| 컬럼 | MAPE | 비고 |
|------|------|------|
| SBRF5.SetPower | 0.39% | 우수 |
| EN4.Power | 0.69% | 우수 |
| PWPDS.Data | 1.05% | 양호 (절대값 수억 단위) |
| Ar.MFC.i | 4.41% | 양호 |
| PLA5.Match.DCBias | 4.63% | 양호 |
| Baratron.Gauge.i | 13.4% | 보통 |
| Ion.Gauge.i | - | 값이 1E-6 수준으로 MAPE 불안정 |

---

## 5. 학습 방법

### 5-1. Aging LSTM 재학습 (현행 학습 스크립트)

| 항목 | 내용 |
|------|------|
| 서버 | `goo4168@bigsoft.time.org -p 7783` |
| 스크립트 경로 | `/home/goo4168/baco/train/train_7feat_aging.py` |

**접속 후 실행:**
```bash
ssh goo4168@bigsoft.time.org -p 7783

cd /home/goo4168/baco/train
python3 train_7feat_aging.py
```

**학습 설정 (스크립트 상단에서 수정 가능):**

```python
FEATURES = [
    'Ar.MFC.i', 'Baratron.Gauge.i', 'PLA5.Match.DCBias',
    'EN4.Power', 'SBRF5.SetPower', 'PWPDS.Data', 'Ion.Gauge.i'
]
INPUT_WINDOW = 10       # 입력 윈도우 (초)
PREDICTION_HORIZON = 5  # 예측 시점 (초)
AUGMENTATION = 50       # 증강 배수
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
EPOCHS = 100
LR = 0.0005
PATIENCE = 15           # Early stopping 기준 에폭
```

**데이터 필터링 규칙:**
- 7개 입력 컬럼이 모두 존재하는 파일만 사용
- 전체 값이 0인 파일 제외
- 최소 길이 (`INPUT_WINDOW + PREDICTION_HORIZON + 1 = 16행`) 미만 제외

**출력 파일 (서버 경로):**
```
/home/goo4168/baco/train/models/PVD4/
├── lstm_7feat_best.pth          # 검증 손실 최소 체크포인트
├── scaler_input_lstm_7feat.pkl  # MinMaxScaler (-1~1)
└── scaler_target_lstm_7feat.pkl
```

**백그라운드 학습 (로그 저장):**
```bash
ssh goo4168@bigsoft.time.org -p 7783

cd /home/goo4168/baco/train
nohup python3 train_7feat_aging.py > /tmp/train_7feat.log 2>&1 &

# 진행 상황 확인
tail -f /tmp/train_7feat.log
```

### 5-2. GPU 환경

| 항목 | 정보 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 × 2 |
| CUDA | 12.7 |
| PyTorch | 2.5.1 |
| 사용 GPU | cuda:0 (기본) |
| 100 에폭 소요 시간 | 약 35분 (2.2M 시퀀스 기준) |

---

## 6. 웹 플랫폼 기능

| 항목 | 내용 |
|------|------|
| 접속 URL | `http://bigsoft.time.org:9300` |
| 서버 코드 경로 | `/home/goo4168/baco/pvd_monitoring/` |

### 6-1. 탭 구성

#### (1) 실시간 모니터링 탭
- 8개 센서 실시간 차트 (Ar.MFC.i, EN4.Power, SBRF5.SetPower, PLA5.Match.DCBias, Tune.Posi, Stage1.Temp, EN4.Volt, Ion.Gauge.i)
- CNN 모델의 PWPDS.Data 예측값 차트 (실제값 vs 예측값 + bounds)
- Ar 유량 진동 구간 하이라이트
- WebSocket `/ws/inference` 로 1초마다 갱신

#### (2) 장비 노후 감지 탭
- Aging LSTM 모델 기반 7개 센서 5초 후 예측 차트
  - Ar.MFC.i, Baratron.Gauge.i, PLA5.Match.DCBias
  - EN4.Power, SBRF5.SetPower, PWPDS.Data, Ion.Gauge.i
- 실제값(파란선) vs 예측값(주황 점선) 동시 표시
- 우측 이상 로그 패널 (컬럼별 오차 10% 이상 발생 시 기록)
- WebSocket `/ws/aging` 로 1초마다 갱신

#### (3) 이상 로그 탭
- 탐지된 이상 구간 목록 (유형·심각도 필터 지원)
- 통계 요약: 총 이상 구간, 유형별/심각도별 카운트
- 이상 유형: AR_FLUCTUATION / PWPDS_PREDICTION_ANOMALY / MAGNET_CONTAMINATION / GAS_FLOW_ANOMALY

#### (4) 공정 이력 탭
- 과거 공정별 데이터 및 이상 구간 조회
- 공정 선택 → 전체 시계열 차트 표시

#### (5) 분석 리포트 탭
- 공정별 상세 분석 리포트

### 6-2. 데이터 흐름

```
CSV 파일 생성 (data/)
        ↓
FileWatcher (abnormal_monitor.py)   ← 0.5초마다 새 행 감지
        ↓
PVDAnomalyDetector                  ← 규칙 기반 이상감지
        ↓ (동시)
CNNInferenceEngine                  ← PWPDS 예측 이상감지
        ↓ (동시)
AgingInferenceEngine                ← 7센서 노후 감지
        ↓
WebSocket 브로드캐스트 → 브라우저 차트 갱신
```

### 6-3. REST API 주요 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 모니터링 상태 |
| GET | `/api/data/realtime?count=N` | 실시간 데이터 (최대 10000행) |
| GET | `/api/intervals` | 이상 구간 목록 |
| GET | `/api/logs` | 이상 로그 목록 |
| GET | `/api/history/{process_id}` | 공정별 전체 데이터 |
| GET | `/api/column-ranges` | 차트 Y축 범위 |
| POST | `/api/generator/start?mode=normal\|abnormal` | 데이터 생성기 시작 |
| POST | `/api/generator/stop` | 데이터 생성기 중지 |
| DELETE | `/api/history/{process_id}` | 공정 데이터 삭제 |

### 6-4. WebSocket 엔드포인트

| 경로 | 설명 | 사용 추론 엔진 |
|------|------|--------------|
| `/ws` | 상태 + 전체 센서 데이터 | 규칙 기반 감지 |
| `/ws/inference` | CNN 추론 결과 | CNNInferenceEngine |
| `/ws/aging` | Aging LSTM 추론 결과 | AgingInferenceEngine |

### 6-5. 데이터 생성기 (테스트용)

`generate_data.py`는 `PDS_Data_Log`의 실제 CSV를 읽어 1초 간격으로 `data/` 디렉토리에 재생:
```
정상 모드  : 원본 데이터 그대로 재생
비정상 모드: PWPDS.Data를 의도적으로 bounds 초과값으로 주입
```

---

## 7. 실행 방법

> 모든 명령은 서버 `goo4168@bigsoft.time.org -p 7783` 에서 실행

### 7-1. 웹 서버 실행

```bash
ssh goo4168@bigsoft.time.org -p 7783

cd /home/goo4168/baco/pvd_monitoring

# 의존성 설치 (최초 1회)
./run.sh install

# 웹 서버 시작 (포트 9300)
./run.sh server
```

접속: `http://bigsoft.time.org:9300`

### 7-2. 데모 모드 (데이터 생성 + 웹 서버 동시)

```bash
ssh goo4168@bigsoft.time.org -p 7783
cd /home/goo4168/baco/pvd_monitoring

# 정상 데이터 생성 + 웹 서버
./run.sh demo

# 비정상 데이터 포함 데모 (이상 주입)
./run.sh generate --inject-anomaly &
./run.sh server
```

### 7-3. 테스트 데이터 생성기만 실행

```bash
ssh goo4168@bigsoft.time.org -p 7783
cd /home/goo4168/baco/pvd_monitoring

./run.sh generate                    # 단일 공정 (정상)
./run.sh generate --continuous       # 연속 생성
./run.sh generate --inject-anomaly   # 이상 데이터 주입
```

### 7-4. Aging 모델 재학습

```bash
ssh goo4168@bigsoft.time.org -p 7783
cd /home/goo4168/baco/train

# 포그라운드 (로그 실시간 출력)
python3 train_7feat_aging.py

# 백그라운드
nohup python3 train_7feat_aging.py > /tmp/train_7feat.log 2>&1 &
tail -f /tmp/train_7feat.log
```

학습 완료 후 웹 서버를 재시작하면 새 모델이 자동 로드됨.

### 7-5. CLI 모니터링 모드

```bash
ssh goo4168@bigsoft.time.org -p 7783
cd /home/goo4168/baco/pvd_monitoring
./run.sh monitor
```

---

## 8. 주요 파일 참조

> **서버:** `goo4168@bigsoft.time.org -p 7783`
> **기본 경로:** `/home/goo4168/baco/`

### 웹 플랫폼 코드

| 파일명 | 서버 전체 경로 | 역할 |
|--------|--------------|------|
| `main.py` | `/home/goo4168/baco/pvd_monitoring/main.py` | FastAPI 앱, 모든 API/WS 라우팅 |
| `abnormal_monitor.py` | `/home/goo4168/baco/pvd_monitoring/abnormal_monitor.py` | CSV 감시 + 3개 추론엔진 조율 |
| `lstm_inference.py` | `/home/goo4168/baco/pvd_monitoring/lstm_inference.py` | LSTM PWPDS 예측 엔진 |
| `cnn_inference.py` | `/home/goo4168/baco/pvd_monitoring/cnn_inference.py` | CNN PWPDS 예측 + bounds 감지 엔진 |
| `aging_inference.py` | `/home/goo4168/baco/pvd_monitoring/aging_inference.py` | Aging 7센서 예측 엔진 |
| `generate_data.py` | `/home/goo4168/baco/pvd_monitoring/generate_data.py` | 테스트 데이터 재생기 |
| `anomaly_detector.py` | `/home/goo4168/baco/pvd_monitoring/models/anomaly_detector.py` | 규칙 기반 이상감지 로직 |
| `index.html` | `/home/goo4168/baco/pvd_monitoring/templates/index.html` | 웹 대시보드 (Chart.js) |
| `run.sh` | `/home/goo4168/baco/pvd_monitoring/run.sh` | 통합 실행 스크립트 |

### 모델 파일

| 파일명 | 서버 전체 경로 | 용도 |
|--------|--------------|------|
| `model_B_aug100x_best.pt` | `/home/goo4168/baco/lstm_project/models/model_B_aug100x_best.pt` | LSTM PWPDS 예측 모델 |
| `model_B_aug100x_scaler_x.pkl` | `/home/goo4168/baco/lstm_project/models/model_B_aug100x_scaler_x.pkl` | LSTM 입력 스케일러 |
| `model_B_aug100x_scaler_y.pkl` | `/home/goo4168/baco/lstm_project/models/model_B_aug100x_scaler_y.pkl` | LSTM 출력 스케일러 |
| `v3_CNN_alldata_aug50x_best.pt` | `/home/goo4168/baco/lstm_project/models/v3_CNN_alldata_aug50x_best.pt` | CNN PWPDS 예측 모델 |
| `v3_CNN_alldata_aug50x_scaler_x.pkl` | `/home/goo4168/baco/lstm_project/models/v3_CNN_alldata_aug50x_scaler_x.pkl` | CNN 입력 스케일러 |
| `v3_CNN_alldata_aug50x_scaler_y.pkl` | `/home/goo4168/baco/lstm_project/models/v3_CNN_alldata_aug50x_scaler_y.pkl` | CNN 출력 스케일러 |
| `bounds_lookup.json` | `/home/goo4168/baco/lstm_project/results/bounds_lookup.json` | 레시피별 PWPDS 상하한 경계값 |
| `column_ranges.json` | `/home/goo4168/baco/lstm_project/results/column_ranges.json` | 차트 Y축 범위값 |
| `lstm_7feat_best.pth` | `/home/goo4168/baco/train/models/PVD4/lstm_7feat_best.pth` | Aging 7센서 예측 모델 |
| `scaler_input_lstm_7feat.pkl` | `/home/goo4168/baco/train/models/PVD4/scaler_input_lstm_7feat.pkl` | Aging 입력 스케일러 |
| `scaler_target_lstm_7feat.pkl` | `/home/goo4168/baco/train/models/PVD4/scaler_target_lstm_7feat.pkl` | Aging 출력 스케일러 |

### 학습 코드

| 파일명 | 서버 전체 경로 | 역할 |
|--------|--------------|------|
| `train_7feat_aging.py` | `/home/goo4168/baco/train/train_7feat_aging.py` | Aging LSTM 학습 스크립트 (현행) |

### 데이터

| 경로 | 서버 전체 경로 | 설명 |
|------|--------------|------|
| `PDS_Data_Log/` | `/home/goo4168/baco/PDS_Data_Log/` | 레시피 실험 데이터 (45개 공정, 1초 간격) |
| `minio_csv/.../PVD4/` | `/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/` | 실제 운용 데이터 (243개 파일) |
| `pvd_monitoring/data/` | `/home/goo4168/baco/pvd_monitoring/data/` | 실시간 모니터링 대상 디렉토리 |
| `backup/` | `/home/goo4168/baco/backup/` | 구버전 모델·코드 보관 |
