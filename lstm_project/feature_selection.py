#!/usr/bin/env python3
"""
PWPDS.Data 예측을 위한 피처 선별 분석
- Pearson/Spearman 상관계수
- Mutual Information (비선형 의존성)
- Random Forest / XGBoost Feature Importance
- 종합 랭킹
"""

import os
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

# ============================================================
# 1. 데이터 로드
# ============================================================
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log/'
TARGET = 'PWPDS.Data'

CANDIDATES = [
    'ULVAC.Stage1.Temp1', 'ULVAC.Stage2.Temp1',
    'EN4.Power', 'EN4.Current', 'EN4.Volt',
    'PLA5.Match.Load.Posi', 'PLA5.Match.Tune.Posi',
    'PLA5.Match.Load.Pre', 'PLA5.Match.Tune.Pre',
    'PLA5.Match.DCBias',
    'SBRF5.Forward', 'SBRF5.Reflect', 'SBRF5.SetPower',
    'PWESC.Volt1', 'PWESC.Volt2',
    'OES.Data6',
    'Line.Gauge.i', 'Ion.Gauge.i', 'Baratron.Gauge.i',
    'Ar.MFC.i', 'Ar2.MFC.i', 'Ar.MFC.o', 'Ar2.MFC.o',
]

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"CSV 파일 수: {len(csv_files)}")

frames = []
for f in csv_files:
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    # 수치형 변환
    for col in CANDIDATES + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    sub = df[CANDIDATES + [TARGET]].dropna()
    if len(sub) > 0:
        frames.append(sub)

data = pd.concat(frames, ignore_index=True)
print(f"전체 활성 구간 데이터: {len(data)} rows\n")

X = data[CANDIDATES]
y = data[TARGET]

# 분산이 0인 컬럼(상수값) 제거
zero_var = X.columns[X.std() == 0].tolist()
if zero_var:
    print(f"[제외] 분산=0 (상수값) 컬럼: {zero_var}")
    CANDIDATES = [c for c in CANDIDATES if c not in zero_var]
    X = X[CANDIDATES]
print(f"분석 대상 피처: {len(CANDIDATES)}개\n")

# ============================================================
# 2. Pearson 상관계수 (선형 관계)
# ============================================================
print("=" * 70)
print("[ 1. Pearson 상관계수 ] — 선형 관계 강도")
print("=" * 70)
pearson = X.corrwith(y).abs().sort_values(ascending=False).dropna()
for col, val in pearson.items():
    bar = '█' * int(val * 40)
    print(f"  {col:<30s} {val:.4f}  {bar}")

# ============================================================
# 3. Spearman 상관계수 (단조 관계, 비선형 포함)
# ============================================================
print(f"\n{'=' * 70}")
print("[ 2. Spearman 상관계수 ] — 단조(monotonic) 관계 강도")
print("=" * 70)
spearman = {}
for col in CANDIDATES:
    rho, _ = stats.spearmanr(X[col], y)
    spearman[col] = abs(rho)
spearman = pd.Series(spearman).sort_values(ascending=False)
for col, val in spearman.items():
    bar = '█' * int(val * 40)
    print(f"  {col:<30s} {val:.4f}  {bar}")

# ============================================================
# 4. Mutual Information (비선형 의존성)
# ============================================================
print(f"\n{'=' * 70}")
print("[ 3. Mutual Information ] — 비선형 의존성 (높을수록 정보량 많음)")
print("=" * 70)
mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi, index=CANDIDATES)
mi_norm = mi_series / mi_series.max()  # 0~1 정규화
mi_norm = mi_norm.sort_values(ascending=False)
for col, val in mi_norm.items():
    bar = '█' * int(val * 40)
    print(f"  {col:<30s} {val:.4f}  {bar}")

# ============================================================
# 5. Random Forest Feature Importance
# ============================================================
print(f"\n{'=' * 70}")
print("[ 4. Random Forest Feature Importance ]")
print("=" * 70)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)
rf_imp = pd.Series(rf.feature_importances_, index=CANDIDATES).sort_values(ascending=False)
for col, val in rf_imp.items():
    bar = '█' * int(val / rf_imp.max() * 40)
    print(f"  {col:<30s} {val:.4f}  {bar}")
print(f"  R² score: {rf.score(X_scaled, y):.4f}")

# ============================================================
# 6. XGBoost Feature Importance (설치되어 있으면)
# ============================================================
xgb_imp = None
try:
    from xgboost import XGBRegressor
    print(f"\n{'=' * 70}")
    print("[ 5. XGBoost Feature Importance ]")
    print("=" * 70)
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                        random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_scaled, y)
    xgb_imp = pd.Series(xgb.feature_importances_, index=CANDIDATES).sort_values(ascending=False)
    for col, val in xgb_imp.items():
        bar = '█' * int(val / xgb_imp.max() * 40)
        print(f"  {col:<30s} {val:.4f}  {bar}")
    print(f"  R² score: {xgb.score(X_scaled, y):.4f}")
except ImportError:
    print("\n[INFO] xgboost 미설치 — XGBoost 분석 생략")

# ============================================================
# 7. 종합 랭킹 (평균 순위 기반)
# ============================================================
print(f"\n{'=' * 70}")
print("[ 종합 랭킹 ] — 모든 방법의 순위 평균 (낮을수록 중요)")
print("=" * 70)

ranks = pd.DataFrame(index=CANDIDATES)
ranks['Pearson'] = pearson.rank(ascending=False)
ranks['Spearman'] = spearman.rank(ascending=False)
ranks['MI'] = mi_norm.rank(ascending=False)
ranks['RF'] = rf_imp.rank(ascending=False)
if xgb_imp is not None:
    ranks['XGB'] = xgb_imp.rank(ascending=False)

ranks['평균순위'] = ranks.mean(axis=1)
ranks = ranks.sort_values('평균순위')

print(f"\n  {'Feature':<30s} ", end='')
for c in ranks.columns:
    print(f"{c:>8s}", end='')
print()
print("  " + "-" * (30 + 8 * len(ranks.columns)))

for col, row in ranks.iterrows():
    print(f"  {col:<30s} ", end='')
    for c in ranks.columns:
        print(f"{row[c]:>8.1f}", end='')
    print()

# 추천 피처
top_n = 8
top_features = ranks.index[:top_n].tolist()
print(f"\n  ★ 추천 TOP-{top_n} 피처: {top_features}")

# ============================================================
# 8. 다중공선성 체크 (VIF)
# ============================================================
print(f"\n{'=' * 70}")
print(f"[ 다중공선성 체크 (VIF) ] — TOP-{top_n} 피처 간 중복 확인")
print("  VIF > 10 이면 다른 피처와 중복 정보가 많음 → 제거 고려")
print("=" * 70)
from sklearn.linear_model import LinearRegression

X_top = data[top_features]
X_top_scaled = StandardScaler().fit_transform(X_top)

for i, col in enumerate(top_features):
    others = np.delete(X_top_scaled, i, axis=1)
    r2 = LinearRegression().fit(others, X_top_scaled[:, i]).score(others, X_top_scaled[:, i])
    vif = 1 / (1 - r2) if r2 < 1 else float('inf')
    flag = " ⚠ 높음" if vif > 10 else ""
    print(f"  {col:<30s} VIF = {vif:>8.2f}{flag}")

print(f"\n{'=' * 70}")
print("[ 완료 ] 위 결과를 참고하여 피처를 선택하세요.")
print("  - 종합 랭킹 상위 피처 중 VIF가 높은 쌍은 하나만 남기는 것을 권장")
print("  - 예: EN4.Power와 EN4.Current가 모두 상위면 하나만 선택")
print("=" * 70)
