# PVD 공정 이상 분석 리포트 구조

> `models/anomaly_detector.py` 기반
> 핵심 메서드: `_analyze_interval()` / `generate_report()`

---

## 목차

1. [메서드 역할 비교](#1-메서드-역할-비교)
2. [_analyze_interval() 분기 흐름](#2-_analyze_interval-분기-흐름)
3. [이상 유형별 분석 상세](#3-이상-유형별-분석-상세)
4. [generate_report() 집계 흐름](#4-generate_report-집계-흐름)
5. [recommendations 생성 기준](#5-recommendations-생성-기준)

---

## 1. 메서드 역할 비교

| 구분 | `_analyze_interval()` | `generate_report()` |
|:---:|:---|:---|
| **호출 시점** | 이상 구간 저장 시 (`add_interval()` 내부) | API 요청 시 (`/api/report/{process_id}`) |
| **입력** | 단일 `AnomalyInterval` 객체 | `process_id` (공정 ID) |
| **출력** | `{ summary, details, cause, impact }` | `{ total, by_type, by_severity, intervals[], recommendations[] }` |
| **범위** | 구간 **1개** 분석 | 공정 **전체** 구간 집계 |
| **관계** | 분석 결과를 구간 데이터에 내장 | 내장된 분석 결과를 그대로 재사용 |

```
이상 구간 저장 시:
  add_interval()
    └── _analyze_interval()  →  { summary, cause, impact } 구간에 내장

API 조회 시:
  generate_report()
    └── 저장된 구간들 집계
          ├── analysis 필드 (이미 내장됨) 그대로 포함
          └── recommendations 새로 생성 (횟수 기반 우선순위)
```

---

## 2. `_analyze_interval()` 분기 흐름

```
_analyze_interval(interval)
│
├── interval.anomaly_type == 'MAGNET_CONTAMINATION'
│     └── → [마그넷 오염 분석]
│
├── interval.anomaly_type == 'GAS_FLOW_ANOMALY'
│     ├── deviation > 0  →  [가스 과다 유입 분석]
│     └── deviation ≤ 0  →  [가스 부족 분석]
│
├── interval.anomaly_type == 'AR_FLUCTUATION'
│     └── → [Ar 유량 진동 분석]
│
└── interval.anomaly_type == 'PWPDS_PREDICTION_ANOMALY'
      ├── actual < predicted  →  [PWPDS 저하 분석]
      └── actual ≥ predicted  →  [PWPDS 상승 분석]

※ 각 분기에서 공통적으로 { summary, details, cause, impact } 반환
```

---

## 3. 이상 유형별 분석 상세

### 3-1. MAGNET_CONTAMINATION — 마그넷 오염

> **감지 조건**: `OES.Data6` 감소 + `EN4.Power`(DC Power) 유지/증가 동시 발생

| 항목 | 내용 |
|:---:|:---|
| **요약** | 마그넷 오염으로 인한 플라즈마 밀도 저하 감지 |
| **상세** | · OES 신호 감소량 (peak 기준) <br> · DC Power 유지/증가 확인 <br> · 이상 지속 포인트 수 |
| **원인** | 마그넷 표면에 오염물질(증착 부산물, 파티클 등) 축적 → 플라즈마 발생 효율 저하 |
| **영향** | 증착 균일도 저하, 막 품질 불량, 공정 반복성 감소 |

---

### 3-2. GAS_FLOW_ANOMALY — 가스 유량 이상

> **감지 조건**: `Ar.MFC.i`가 안정 베이스라인 대비 **±1.5% 이상** 변동

#### Case A. 가스 과다 유입 (`deviation > 0`)

| 항목 | 내용 |
|:---:|:---|
| **요약** | 가스 과다 유입 감지 (기준값 대비 +N%) |
| **상세** | · Ar.MFC.i 기준값 (sccm) <br> · 최대 편차 (+N%) <br> · 이상 지속 포인트 수 |
| **원인** | MFC 제어 오류, 가스 라인 압력 변동, 또는 밸브 오작동 가능성 |
| **영향** | 증착률 변화, 막 조성 변화, 챔버 압력 불안정 |

#### Case B. 가스 부족 (`deviation ≤ 0`)

| 항목 | 내용 |
|:---:|:---|
| **요약** | 가스 부족 감지 (기준값 대비 -N%) |
| **상세** | · Ar.MFC.i 기준값 (sccm) <br> · 최대 편차 (-N%) <br> · 이상 지속 포인트 수 |
| **원인** | 가스 공급 라인 막힘, MFC 고장, 또는 가스 소스 압력 저하 가능성 |
| **영향** | 플라즈마 불안정, 증착률 저하, 막 품질 저하 |

---

### 3-3. AR_FLUCTUATION — Ar 유량 진동

> **감지 조건**: ±1% 초과 진동이 **4초 이상** 지속 + **Zero-crossing 2회 이상** (단순 step change가 아닌 진동임을 확인)

| 항목 | 내용 |
|:---:|:---|
| **요약** | Ar 유량 진동 감지 (±N%, N초 지속) |
| **상세** | · Ar.MFC.i 기준값 (sccm) <br> · 최대 편차 (±N%) <br> · Zero-crossing 횟수 <br> · 이상 지속 포인트 수 |
| **원인** | MFC 제어 오류 → 방전 불안정 / overpressure → 이온화 변동 |
| **영향** | Ar 가스 유량 불안정으로 인한 플라즈마 밀도 변동, 증착 균일도 저하 |

---

### 3-4. PWPDS_PREDICTION_ANOMALY — PWPDS 예측 이상

> **감지 조건**: CNN 예측값 대비 실측값이 **5% 이상** (warning) / **10% 이상** (critical) 차이

#### Case A. 실측값 < 예측값 (저하)

| 항목 | 내용 |
|:---:|:---|
| **요약** | PWPDS.Data 예측 대비 저하 감지 (오차: N%) |
| **상세** | · 실제값 vs 예측값 수치 <br> · 최대 오차 (%) <br> · 이상 지속 포인트 수 |
| **원인** | 마그넷 오염: 금속입자 / 산화물 / 잔류타겟 → 자기장↓ → 전자궤도유지력↓ → 플라즈마 밀도↓ → 스퍼터링 속도↓ |
| **영향** | PWPDS.Data(플라즈마 밀도) 감소 → 증착 균일도 저하 |

#### Case B. 실측값 > 예측값 (상승)

| 항목 | 내용 |
|:---:|:---|
| **요약** | PWPDS.Data 예측 대비 상승 감지 (오차: N%) |
| **상세** | · 실제값 vs 예측값 수치 <br> · 최대 오차 (%) <br> · 이상 지속 포인트 수 |
| **원인** | 타겟-웨이퍼 거리 감소 → 타겟 근처 고밀도 플라즈마 → 거리↓ → 밀도↑ → edge 증착 불균일 |
| **영향** | PWPDS.Data(플라즈마 밀도) 증가 → edge 증착 불균일 |

---

## 4. `generate_report()` 집계 흐름

```
generate_report(process_id)
│
├── 해당 process_id 이상 구간 전체 조회
│
├── 구간 없음?
│     └── { summary: '이상 없음' } 반환 후 종료
│
└── 구간 있음 → 반복 순회
      │
      ├── by_type[anomaly_type]    += 1   (유형별 카운트)
      ├── by_severity[severity]    += 1   (심각도별 카운트)
      └── intervals[].append({
              id, anomaly_type, severity,
              start/end_timestamp,
              duration_points,
              analysis: { summary, details, cause, impact },  ← _analyze_interval() 결과 재사용
              peak_values,
              data_points
          })
      │
      └── recommendations 생성 (by_type 카운트 기반)
            └── [최종 반환]
```

**최종 반환 구조**

```json
{
  "process_id": "...",
  "total_intervals": N,
  "total_anomalies": N,
  "by_type": {
    "MAGNET_CONTAMINATION": N,
    "GAS_FLOW_ANOMALY": N,
    "AR_FLUCTUATION": N,
    "PWPDS_PREDICTION_ANOMALY": N
  },
  "by_severity": { "warning": N, "critical": N },
  "intervals": [
    {
      "id": 1,
      "anomaly_type": "...",
      "severity": "warning | critical",
      "start_timestamp": "...",
      "end_timestamp": "...",
      "duration_points": N,
      "analysis": { "summary": "...", "details": [...], "cause": "...", "impact": "..." },
      "peak_values": { ... },
      "data_points": [ ... ]
    }
  ],
  "recommendations": [
    { "issue": "...", "action": "...", "priority": "high | medium" }
  ]
}
```

---

## 5. `recommendations` 생성 기준

| 이상 유형 | 조치방안 | 우선순위 기준 |
|:---:|:---|:---:|
| **MAGNET_CONTAMINATION** | 마그넷 상태 점검 및 클리닝 필요. 플라즈마 밀도 저하로 인한 증착 품질 저하 우려. | count **> 2** → `high` <br> count **≤ 2** → `medium` |
| **GAS_FLOW_ANOMALY** | MFC(Mass Flow Controller) 점검 필요. 가스 라인 누수 또는 밸브 고장 가능성 확인. | count **> 2** → `high` <br> count **≤ 2** → `medium` |
| **AR_FLUCTUATION** | MFC 제어 안정성 점검 필요. 방전 불안정 / overpressure로 인한 이온화 변동 우려. | count **> 2** → `high` <br> count **≤ 2** → `medium` |
| **PWPDS_PREDICTION_ANOMALY** | CNN 예측값 대비 실측값 이상 감지. 마그넷 오염 또는 타겟-웨이퍼 거리 변화 확인 필요. | count **> 1** → `high` <br> count **≤ 1** → `medium` |

> **참고**: `PWPDS_PREDICTION_ANOMALY`는 다른 유형보다 임계 횟수가 낮음 (2회 이상이면 high).
> 플라즈마 밀도 직접 이상으로 공정 영향이 즉각적이기 때문.
