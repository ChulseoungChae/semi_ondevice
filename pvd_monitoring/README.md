# PVD 공정 실시간 이상감지 웹 플랫폼

반도체 PVD(Physical Vapor Deposition) 증착 장비의 센서 데이터를 실시간으로 모니터링하고 이상을 감지하는 웹 플랫폼입니다.

## 주요 기능

### 1. 실시간 이상 탐지
- **마그넷 오염 감지**: OES(Optical Emission Spectroscopy) 신호와 DC Power의 상관관계 분석
  - OES 방출선이 감소하면서 DC Power가 유지/증가하면 이상으로 판단
- **가스 유량 이상 감지**: Ar.MFC.i 가스 유량 모니터링
  - 안정 상태에서 ±1.5% 이상 변동 시 이상으로 판단

### 2. 웹 대시보드
- 실시간 센서 데이터 차트 (DC Power, OES, 가스 유량, 진공도)
- 이상 발생 시 즉시 알림
- 이상 로그 이력 검색 및 조회
- 공정별 분석 리포트 생성

### 3. 데이터 생성기
- 원본 데이터 기반 실시간 시뮬레이션
- 테스트를 위한 이상 데이터 주입 기능

## 설치

```bash
cd /home/goo4168/baco/pvd_monitoring

# 의존성 설치
pip install -r requirements.txt
# 또는
./run.sh install
```

## 실행 방법

### 웹 서버 실행
```bash
./run.sh server
# 또는
python main.py
```
브라우저에서 http://localhost:8000 접속

### 데이터 생성기 실행
```bash
# 단일 공정 데이터 생성
./run.sh generate

# 연속 데이터 생성
./run.sh generate --continuous

# 이상 데이터 포함 생성
./run.sh generate --inject-anomaly --anomaly-type magnet
./run.sh generate --inject-anomaly --anomaly-type gas
```

### 데모 모드 (데이터 생성 + 웹 서버)
```bash
./run.sh demo
```

## 프로젝트 구조

```
pvd_monitoring/
├── main.py                 # FastAPI 웹 애플리케이션
├── generate_data.py        # 실시간 데이터 생성기
├── abnormal_monitor.py     # 이상감지 모니터
├── run.sh                  # 실행 스크립트
├── requirements.txt        # 의존성 목록
├── models/
│   ├── __init__.py
│   └── anomaly_detector.py # 이상 탐지 모델
├── data/                   # 생성된 데이터 저장
├── logs/                   # 이상 로그 저장
├── static/                 # 정적 파일
└── templates/
    └── index.html          # 웹 UI
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 메인 대시보드 |
| GET | `/api/status` | 모니터링 상태 |
| GET | `/api/data/realtime` | 실시간 데이터 조회 |
| GET | `/api/logs` | 이상 로그 검색 |
| GET | `/api/logs/summary` | 로그 요약 통계 |
| GET | `/api/report/{process_id}` | 공정별 리포트 |
| GET | `/api/processes` | 공정 목록 |
| WS | `/ws` | 실시간 데이터 스트리밍 |

## 이상 탐지 알고리즘

### 1. 마그넷 오염 감지 (MAGNET_CONTAMINATION)
```
조건: OES.Data6 감소 AND EN4.Power 유지/증가
- 이동 평균 대비 OES 5% 이상 감소
- 동시에 Power가 감소하지 않음
심각도:
- Warning: OES 5~10% 감소
- Critical: OES 10% 이상 감소
```

### 2. 가스 유량 이상 감지 (GAS_FLOW_ANOMALY)
```
조건: Ar.MFC.i 안정 상태에서 ±1.5% 이상 변동
- 연속 5초 이상 값이 안정된 후 기준선 설정
- 기준선 대비 변동률 계산
심각도:
- Warning: ±1.5~3% 변동
- Critical: ±3% 이상 변동
```

## 데이터 형식

### 입력 CSV 형식
```csv
Timer,ULVAC.Stage1.Temp1,EN4.Power,OES.Data6,Ar.MFC.i,...
[2025.01.22 10:30:00],65.0,1000,1.4,100,...
```

### 지원 파일 패턴
- `PVD4_NEW_YYYYMMDD_HHMMSS.csv` - OES 데이터 없음
- `PVD4_YYMMDD_YYYYMMDD_HHMMSS.csv` - OES 데이터 포함

## 라이선스

Internal Use Only
