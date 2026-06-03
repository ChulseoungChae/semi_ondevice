"""
PVD 공정 이상 탐지 모델

이상징후 1: 마그넷 오염 감지
- OES.Data6가 감소하면서 DC Power(EN4.Power)가 증가하면 이상

이상징후 2: 가스 유량 이상 감지
- Ar.MFC.i가 일정값 유지 중 ±1.5% 이상 변동하면 이상

이상징후 3: Ar 유량 진동 감지 (AR_FLUCTUATION)
- 안정구간에서 ±1% 초과 진동이 4초 이상 지속 시 이상
- Zero-crossing 분석: 기준값 위/아래를 번갈아가면 진동

이상징후 4: PWPDS 예측 이상 (PWPDS_PREDICTION_ANOMALY)
- CNN 예측값 대비 5%/10% 이상 차이
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class AnomalyResult:
    """이상 탐지 결과"""
    timestamp: str
    anomaly_type: str
    severity: str  # 'warning', 'critical'
    description: str
    related_values: Dict


@dataclass
class AnomalyInterval:
    """이상 구간 정보"""
    interval_id: int
    anomaly_type: str
    severity: str
    start_timestamp: str
    end_timestamp: Optional[str]
    start_index: int
    end_index: Optional[int]
    description: str
    data_points: List[Dict]  # 구간 내 데이터 포인트들
    peak_values: Dict  # 구간 내 최대/최소 값
    is_active: bool = True  # 진행 중인 이상인지


class PVDAnomalyDetector:
    """PVD 공정 이상 탐지 클래스"""

    def __init__(self, window_size: int = 10, gas_threshold: float = 0.015):
        """
        Args:
            window_size: 이동 평균 계산을 위한 윈도우 크기
            gas_threshold: 가스 유량 변동 임계값 (기본 1.5%)
        """
        self.window_size = window_size
        self.gas_threshold = gas_threshold

        # 이전 데이터 저장 (슬라이딩 윈도우)
        self.oes_history: List[float] = []
        self.power_history: List[float] = []
        self.ar_mfc_history: List[float] = []
        self.ar_mfc_baseline: Optional[float] = None
        self.ar_mfc_stable_count: int = 0

        # 가스 안정 상태 판단 기준 (연속 n개 이상 비슷한 값)
        self.gas_stable_threshold = 5

        # Ar 유량 진동 감지용
        self.ar_fluctuation_threshold = 0.01  # ±1%
        self.ar_fluctuation_min_duration = 4   # 4초 이상 지속
        self.ar_fluctuation_history: List[float] = []
        self.ar_fluctuation_baseline: Optional[float] = None
        self.ar_fluctuation_count: int = 0  # 연속 진동 포인트 수
        self.ar_zero_crossings: int = 0     # zero-crossing 횟수
        self.ar_last_sign: Optional[int] = None  # 마지막 편차 부호

        # 이상 구간 관리
        self.active_intervals: Dict[str, AnomalyInterval] = {}  # anomaly_type -> active interval
        self.pending_intervals: Dict[str, AnomalyInterval] = {}  # 3초 미만 대기 중인 구간
        self.completed_intervals: List[AnomalyInterval] = []
        self.interval_id_counter = 0
        self.data_index = 0  # 현재 데이터 인덱스
        self.min_interval_duration = 3  # 최소 이상 구간 지속 시간 (초/포인트)

    def reset(self):
        """새 공정 시작 시 히스토리 초기화"""
        self.oes_history = []
        self.power_history = []
        self.ar_mfc_history = []
        self.ar_mfc_baseline = None
        self.ar_mfc_stable_count = 0
        self.ar_fluctuation_history = []
        self.ar_fluctuation_baseline = None
        self.ar_fluctuation_count = 0
        self.ar_zero_crossings = 0
        self.ar_last_sign = None
        # 진행 중인 이상 구간 종료
        for interval in self.active_intervals.values():
            interval.is_active = False
            self.completed_intervals.append(interval)
        self.active_intervals = {}
        self.pending_intervals = {}  # 대기 중인 구간도 초기화
        self.data_index = 0

    def get_all_intervals(self) -> List[AnomalyInterval]:
        """모든 확정된 이상 구간 반환 (진행 중 + 완료, 대기 중 제외)"""
        return self.completed_intervals + list(self.active_intervals.values())

    def get_interval_count(self) -> int:
        """확정된 이상 구간 수 반환 (대기 중 제외)"""
        return len(self.completed_intervals) + len(self.active_intervals)

    def get_pending_count(self) -> int:
        """대기 중인 이상 구간 수 반환"""
        return len(self.pending_intervals)

    def detect(self, data: Dict) -> Tuple[List[AnomalyResult], List[AnomalyInterval], List[AnomalyInterval]]:
        """
        단일 데이터 포인트에 대해 이상 탐지 수행
        이상 구간은 3초(3 데이터 포인트) 이상 지속되어야 유효함

        Args:
            data: 센서 데이터 딕셔너리

        Returns:
            (탐지된 이상 목록, 새로 확정된 구간 목록, 종료된 구간 목록)
        """
        anomalies = []
        new_intervals = []  # 3초 이상 지속되어 확정된 새 구간
        ended_intervals = []
        timestamp = data.get('Timer', datetime.now().strftime('[%Y.%m.%d %H:%M:%S]'))

        # 각 이상 유형별 현재 상태 체크
        current_anomaly_types = set()

        # OES 데이터가 있는 경우 마그넷 오염 검사
        if 'OES.Data6' in data:
            oes_anomaly = self._check_magnet_contamination(data, timestamp)
            if oes_anomaly:
                anomalies.append(oes_anomaly)
                current_anomaly_types.add('MAGNET_CONTAMINATION')

        # 가스 유량 이상 검사
        gas_anomaly = self._check_gas_flow_anomaly(data, timestamp)
        if gas_anomaly:
            anomalies.append(gas_anomaly)
            current_anomaly_types.add('GAS_FLOW_ANOMALY')

        # Ar 유량 진동 검사
        ar_fluct_anomaly = self._check_ar_fluctuation(data, timestamp)
        if ar_fluct_anomaly:
            anomalies.append(ar_fluct_anomaly)
            current_anomaly_types.add('AR_FLUCTUATION')

        # 이상 구간 관리
        for anomaly in anomalies:
            atype = anomaly.anomaly_type

            if atype in self.active_intervals:
                # 이미 확정된 구간 업데이트
                interval = self.active_intervals[atype]
                interval.end_timestamp = timestamp
                interval.end_index = self.data_index
                interval.data_points.append(data.copy())
                self._update_peak_values(interval, data)
                if anomaly.severity == 'critical':
                    interval.severity = 'critical'

            elif atype in self.pending_intervals:
                # 대기 중인 구간 업데이트
                interval = self.pending_intervals[atype]
                interval.end_timestamp = timestamp
                interval.end_index = self.data_index
                interval.data_points.append(data.copy())
                self._update_peak_values(interval, data)
                if anomaly.severity == 'critical':
                    interval.severity = 'critical'

                # 3초 이상 지속되면 확정
                duration = len(interval.data_points)
                if duration >= self.min_interval_duration:
                    # 대기에서 활성으로 승격
                    del self.pending_intervals[atype]
                    self.active_intervals[atype] = interval
                    new_intervals.append(interval)
                    print(f"[INFO] 이상 구간 확정: {atype} ({duration}초 지속)")

            else:
                # 새 대기 구간 시작
                self.interval_id_counter += 1
                interval = AnomalyInterval(
                    interval_id=self.interval_id_counter,
                    anomaly_type=atype,
                    severity=anomaly.severity,
                    start_timestamp=timestamp,
                    end_timestamp=timestamp,
                    start_index=self.data_index,
                    end_index=self.data_index,
                    description=anomaly.description,
                    data_points=[data.copy()],
                    peak_values=anomaly.related_values.copy(),
                    is_active=True
                )
                self.pending_intervals[atype] = interval

        # 정상으로 돌아온 이상 유형 처리
        # 1. 대기 중인 구간이 3초 미만으로 끝나면 폐기
        for atype in list(self.pending_intervals.keys()):
            if atype not in current_anomaly_types:
                discarded = self.pending_intervals.pop(atype)
                duration = len(discarded.data_points)
                print(f"[INFO] 이상 구간 폐기 (3초 미만): {atype} ({duration}초)")

        # 2. 확정된 구간이 종료되면 완료 목록에 추가
        for atype in list(self.active_intervals.keys()):
            if atype not in current_anomaly_types:
                interval = self.active_intervals.pop(atype)
                interval.is_active = False
                self.completed_intervals.append(interval)
                ended_intervals.append(interval)

        self.data_index += 1
        return anomalies, new_intervals, ended_intervals

    def _update_peak_values(self, interval: AnomalyInterval, data: Dict):
        """구간 내 peak 값 업데이트"""
        for key, value in interval.peak_values.items():
            if key in data and isinstance(data[key], (int, float)):
                if isinstance(value, (int, float)):
                    if abs(data[key]) > abs(value):
                        interval.peak_values[key] = data[key]

    def _check_magnet_contamination(self, data: Dict, timestamp: str) -> Optional[AnomalyResult]:
        """
        마그넷 오염 감지: OES.Data6가 DC Power와 반비례 관계인지 확인

        정상: OES와 Power가 비례 (둘 다 증가하거나 둘 다 감소)
        이상: OES 감소 + Power 증가 또는 유지
        """
        oes_value = float(data.get('OES.Data6', 0))
        power_value = float(data.get('EN4.Power', 0))

        self.oes_history.append(oes_value)
        self.power_history.append(power_value)

        # 윈도우 크기 유지
        if len(self.oes_history) > self.window_size:
            self.oes_history.pop(0)
            self.power_history.pop(0)

        # 충분한 데이터가 쌓일 때까지 대기
        if len(self.oes_history) < self.window_size:
            return None

        # Power가 0이면 대기 상태이므로 검사 불필요
        if power_value < 100:
            return None

        # OES 변화율과 Power 변화율 계산
        oes_change = self.oes_history[-1] - np.mean(self.oes_history[:-1])
        power_change = self.power_history[-1] - np.mean(self.power_history[:-1])

        # OES 평균값
        oes_mean = np.mean(self.oes_history[:-1])

        # OES가 0이 아니고, 감소하는 동안 Power가 증가하면 이상
        if oes_mean > 0.1 and oes_change < -0.05 * oes_mean and power_change >= 0:
            severity = 'critical' if oes_change < -0.1 * oes_mean else 'warning'
            return AnomalyResult(
                timestamp=timestamp,
                anomaly_type='MAGNET_CONTAMINATION',
                severity=severity,
                description=f'마그넷 오염 의심: OES 신호 감소({oes_change:.4f}) 중 DC Power 유지/증가',
                related_values={
                    'OES.Data6': oes_value,
                    'OES_change': oes_change,
                    'EN4.Power': power_value,
                    'Power_change': power_change
                }
            )
        return None

    def _check_gas_flow_anomaly(self, data: Dict, timestamp: str) -> Optional[AnomalyResult]:
        """
        가스 유량 이상 감지: Ar.MFC.i가 안정 상태에서 ±1.5% 이상 변동

        - 가스가 0에서 급상승 후 일정하게 유지되어야 정상
        - 안정 상태에서 ±1.5% 이상 변동하면 이상
        """
        # Ar.MFC.i 또는 Ar.200.MFC.i 컬럼 확인
        ar_mfc = None
        for key in ['Ar.MFC.i', 'Ar.200.MFC.i']:
            if key in data:
                ar_mfc = float(data[key])
                break

        if ar_mfc is None:
            return None

        self.ar_mfc_history.append(ar_mfc)

        # 윈도우 크기 유지
        if len(self.ar_mfc_history) > self.window_size * 2:
            self.ar_mfc_history.pop(0)

        # 가스가 0이면 대기 상태
        if ar_mfc < 1:
            self.ar_mfc_baseline = None
            self.ar_mfc_stable_count = 0
            return None

        # 안정 상태 판단: 연속으로 비슷한 값이 유지되는지 확인
        if len(self.ar_mfc_history) >= self.gas_stable_threshold:
            recent = self.ar_mfc_history[-self.gas_stable_threshold:]
            recent_std = np.std(recent)
            recent_mean = np.mean(recent)

            # 표준편차가 평균의 0.5% 이내면 안정 상태로 판단
            if recent_mean > 0 and recent_std / recent_mean < 0.005:
                if self.ar_mfc_baseline is None:
                    self.ar_mfc_baseline = recent_mean
                    self.ar_mfc_stable_count = self.gas_stable_threshold
                else:
                    self.ar_mfc_stable_count += 1

        # 베이스라인이 설정된 상태에서 변동 확인
        if self.ar_mfc_baseline is not None and self.ar_mfc_stable_count >= self.gas_stable_threshold:
            deviation = (ar_mfc - self.ar_mfc_baseline) / self.ar_mfc_baseline

            if abs(deviation) >= self.gas_threshold:
                if deviation > 0:
                    description = f'가스 과다 유입: Ar.MFC.i가 기준값({self.ar_mfc_baseline:.1f}) 대비 +{deviation*100:.2f}% 증가'
                    severity = 'critical' if deviation > 0.03 else 'warning'
                else:
                    description = f'가스 부족: Ar.MFC.i가 기준값({self.ar_mfc_baseline:.1f}) 대비 {deviation*100:.2f}% 감소'
                    severity = 'critical' if deviation < -0.03 else 'warning'

                return AnomalyResult(
                    timestamp=timestamp,
                    anomaly_type='GAS_FLOW_ANOMALY',
                    severity=severity,
                    description=description,
                    related_values={
                        'Ar.MFC.i': ar_mfc,
                        'baseline': self.ar_mfc_baseline,
                        'deviation_percent': deviation * 100
                    }
                )
        return None

    def _check_ar_fluctuation(self, data: Dict, timestamp: str) -> Optional[AnomalyResult]:
        """
        Ar 유량 진동 감지: 안정구간에서 ±1% 초과 진동이 4초 이상 지속

        Zero-crossing 분석:
        - 기준값 위/아래를 번갈아가면 진동 (oscillation)
        - 한 방향으로 이동 후 유지면 정상 (step change)
        """
        ar_mfc = None
        for key in ['Ar.MFC.i', 'Ar.200.MFC.i']:
            if key in data:
                ar_mfc = float(data[key])
                break

        if ar_mfc is None:
            return None

        self.ar_fluctuation_history.append(ar_mfc)

        # 히스토리 크기 제한
        if len(self.ar_fluctuation_history) > 30:
            self.ar_fluctuation_history.pop(0)

        # 가스가 0이면 대기 상태
        if ar_mfc < 1:
            self.ar_fluctuation_baseline = None
            self.ar_fluctuation_count = 0
            self.ar_zero_crossings = 0
            self.ar_last_sign = None
            return None

        # 안정 베이스라인 설정 (최근 10개 평균, std < 0.5%)
        if len(self.ar_fluctuation_history) >= 10:
            recent_10 = self.ar_fluctuation_history[-10:]
            mean_10 = np.mean(recent_10)
            std_10 = np.std(recent_10)

            if self.ar_fluctuation_baseline is None:
                if mean_10 > 0 and std_10 / mean_10 < 0.005:
                    self.ar_fluctuation_baseline = mean_10

        if self.ar_fluctuation_baseline is None:
            return None

        # 편차 계산
        deviation = (ar_mfc - self.ar_fluctuation_baseline) / self.ar_fluctuation_baseline

        # ±1% 초과 진동 체크
        if abs(deviation) > self.ar_fluctuation_threshold:
            # Zero-crossing 분석
            current_sign = 1 if deviation > 0 else -1
            if self.ar_last_sign is not None and current_sign != self.ar_last_sign:
                self.ar_zero_crossings += 1
            self.ar_last_sign = current_sign
            self.ar_fluctuation_count += 1

            # 4초 이상 지속 + 2회 이상 zero-crossing (진동 확인)
            if self.ar_fluctuation_count >= self.ar_fluctuation_min_duration and self.ar_zero_crossings >= 2:
                return AnomalyResult(
                    timestamp=timestamp,
                    anomaly_type='AR_FLUCTUATION',
                    severity='warning',
                    description=f'Ar 유량 진동: 기준값({self.ar_fluctuation_baseline:.1f}) 대비 ±{abs(deviation)*100:.2f}% 진동 ({self.ar_fluctuation_count}초 지속, {self.ar_zero_crossings}회 교차)',
                    related_values={
                        'Ar.MFC.i': ar_mfc,
                        'baseline': self.ar_fluctuation_baseline,
                        'deviation_percent': deviation * 100,
                        'duration': self.ar_fluctuation_count,
                        'zero_crossings': self.ar_zero_crossings,
                    }
                )
        else:
            # 정상 범위로 복귀
            self.ar_fluctuation_count = 0
            self.ar_zero_crossings = 0
            self.ar_last_sign = None

        return None


class AnomalyLogger:
    """이상 로그 관리 클래스 (구간 기반)"""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'anomaly_logs.json')
        self.interval_file = os.path.join(log_dir, 'anomaly_intervals.json')
        self.process_data_dir = os.path.join(log_dir, 'process_data')
        os.makedirs(self.process_data_dir, exist_ok=True)
        self.logs: List[Dict] = self._load_logs()
        self.intervals: List[Dict] = self._load_intervals()

    def _load_logs(self) -> List[Dict]:
        """기존 로그 파일 로드"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _load_intervals(self) -> List[Dict]:
        """기존 구간 파일 로드"""
        if os.path.exists(self.interval_file):
            try:
                with open(self.interval_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def add_log(self, anomaly: AnomalyResult, process_id: str) -> Dict:
        """이상 로그 추가 (개별 포인트 - 하위 호환용)"""
        log_entry = {
            'id': len(self.logs) + 1,
            'process_id': process_id,
            'timestamp': anomaly.timestamp,
            'anomaly_type': anomaly.anomaly_type,
            'severity': anomaly.severity,
            'description': anomaly.description,
            'related_values': anomaly.related_values,
            'created_at': datetime.now().isoformat()
        }
        self.logs.append(log_entry)
        self._save_logs()
        return log_entry

    def add_interval(self, interval: AnomalyInterval, process_id: str) -> Dict:
        """이상 구간 추가"""
        interval_entry = {
            'id': interval.interval_id,
            'process_id': process_id,
            'anomaly_type': interval.anomaly_type,
            'severity': interval.severity,
            'start_timestamp': interval.start_timestamp,
            'end_timestamp': interval.end_timestamp,
            'start_index': interval.start_index,
            'end_index': interval.end_index,
            'duration_points': (interval.end_index or interval.start_index) - interval.start_index + 1,
            'description': interval.description,
            'peak_values': interval.peak_values,
            'data_points': interval.data_points,
            'analysis': self._analyze_interval(interval),
            'created_at': datetime.now().isoformat()
        }
        # 기존 구간 업데이트 또는 추가
        existing_idx = None
        for i, inv in enumerate(self.intervals):
            if inv['process_id'] == process_id and inv['id'] == interval.interval_id:
                existing_idx = i
                break
        if existing_idx is not None:
            self.intervals[existing_idx] = interval_entry
        else:
            self.intervals.append(interval_entry)
        self._save_intervals()
        return interval_entry

    def _analyze_interval(self, interval: AnomalyInterval) -> Dict:
        """이상 구간 분석 - 무엇이 잘못되었는지 설명 생성"""
        analysis = {
            'summary': '',
            'details': [],
            'cause': '',
            'impact': ''
        }

        if interval.anomaly_type == 'MAGNET_CONTAMINATION':
            oes_change = interval.peak_values.get('OES_change', 0)
            power_change = interval.peak_values.get('Power_change', 0)

            analysis['summary'] = '마그넷 오염으로 인한 플라즈마 밀도 저하 감지'
            analysis['details'] = [
                f"OES 신호가 {abs(oes_change):.4f} 만큼 감소하였으나 DC Power는 유지/증가",
                "정상 상태에서는 OES와 DC Power가 비례 관계를 유지해야 함",
                f"구간 내 {len(interval.data_points)}개 데이터 포인트에서 이상 지속"
            ]
            analysis['cause'] = '마그넷 표면에 오염물질(증착 부산물, 파티클 등) 축적으로 플라즈마 발생 효율 저하'
            analysis['impact'] = '증착 균일도 저하, 막 품질 불량, 공정 반복성 감소 가능성'

        elif interval.anomaly_type == 'GAS_FLOW_ANOMALY':
            deviation = interval.peak_values.get('deviation_percent', 0)
            baseline = interval.peak_values.get('baseline', 0)

            if deviation > 0:
                analysis['summary'] = f'가스 과다 유입 감지 (기준값 대비 +{deviation:.2f}%)'
                analysis['details'] = [
                    f"Ar.MFC.i 기준값: {baseline:.1f} sccm",
                    f"최대 편차: +{deviation:.2f}%",
                    f"구간 내 {len(interval.data_points)}개 데이터 포인트에서 이상 지속"
                ]
                analysis['cause'] = 'MFC 제어 오류, 가스 라인 압력 변동, 또는 밸브 오작동 가능성'
                analysis['impact'] = '증착률 변화, 막 조성 변화, 챔버 압력 불안정 가능성'
            else:
                analysis['summary'] = f'가스 부족 감지 (기준값 대비 {deviation:.2f}%)'
                analysis['details'] = [
                    f"Ar.MFC.i 기준값: {baseline:.1f} sccm",
                    f"최대 편차: {deviation:.2f}%",
                    f"구간 내 {len(interval.data_points)}개 데이터 포인트에서 이상 지속"
                ]
                analysis['cause'] = '가스 공급 라인 막힘, MFC 고장, 또는 가스 소스 압력 저하 가능성'
                analysis['impact'] = '플라즈마 불안정, 증착률 저하, 막 품질 저하 가능성'

        elif interval.anomaly_type == 'AR_FLUCTUATION':
            baseline = interval.peak_values.get('baseline', 0)
            deviation = interval.peak_values.get('deviation_percent', 0)
            zero_crossings = interval.peak_values.get('zero_crossings', 0)
            duration = interval.peak_values.get('duration', 0)

            analysis['summary'] = f'Ar 유량 진동 감지 (±{abs(deviation):.2f}%, {duration}초 지속)'
            analysis['details'] = [
                f"Ar.MFC.i 기준값: {baseline:.1f} sccm",
                f"최대 편차: ±{abs(deviation):.2f}%",
                f"Zero-crossing 횟수: {zero_crossings}회 (진동 확인)",
                f"구간 내 {len(interval.data_points)}개 데이터 포인트에서 이상 지속"
            ]
            analysis['cause'] = 'MFC 제어 오류 → 방전 불안정/overpressure → 이온화 변동'
            analysis['impact'] = 'Ar 가스 유량 불안정으로 인한 플라즈마 밀도 변동, 증착 균일도 저하'

        elif interval.anomaly_type == 'PWPDS_PREDICTION_ANOMALY':
            error_pct = interval.peak_values.get('error_pct', 0)
            actual = interval.peak_values.get('actual_pwpds', 0)
            predicted = interval.peak_values.get('predicted_pwpds', 0)

            if actual < predicted:
                analysis['summary'] = f'PWPDS.Data 예측 대비 저하 감지 (오차: {error_pct:.2f}%)'
                analysis['details'] = [
                    f"실제값이 예측값보다 낮음: 실제={actual:.0f}, 예측={predicted:.0f}",
                    f"최대 오차: {error_pct:.2f}%",
                    f"구간 내 {len(interval.data_points)}개 데이터 포인트에서 이상 지속"
                ]
                analysis['cause'] = '마그넷 오염: 금속입자/산화물/잔류타겟 → 자기장↓ → 전자궤도유지력↓ → 플라즈마밀도↓ → 스퍼터링속도↓'
                analysis['impact'] = 'PWPDS.Data(플라즈마 밀도) 감소로 증착 균일도 저하'
            else:
                analysis['summary'] = f'PWPDS.Data 예측 대비 상승 감지 (오차: {error_pct:.2f}%)'
                analysis['details'] = [
                    f"실제값이 예측값보다 높음: 실제={actual:.0f}, 예측={predicted:.0f}",
                    f"최대 오차: {error_pct:.2f}%",
                    f"구간 내 {len(interval.data_points)}개 데이터 포인트에서 이상 지속"
                ]
                analysis['cause'] = '타겟-웨이퍼 거리 감소: 타겟 근처 고밀도 플라즈마 → 거리↓ → 밀도↑ → edge 증착 불균일'
                analysis['impact'] = 'PWPDS.Data(플라즈마 밀도) 증가로 edge 증착 불균일'

        return analysis

    def _save_logs(self):
        """로그 파일 저장"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def _save_intervals(self):
        """구간 파일 저장"""
        with open(self.interval_file, 'w', encoding='utf-8') as f:
            json.dump(self.intervals, f, ensure_ascii=False, indent=2)

    def save_process_data(self, process_id: str, data: Dict):
        """공정 데이터 저장 (이력 조회용)"""
        filepath = os.path.join(self.process_data_dir, f'{process_id}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[INFO] 공정 데이터 저장: {filepath}")

    def get_process_data(self, process_id: str) -> Optional[Dict]:
        """저장된 공정 데이터 조회"""
        filepath = os.path.join(self.process_data_dir, f'{process_id}.json')
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ERROR] 공정 데이터 로드 실패: {e}")
        return None

    def get_all_process_ids(self) -> List[str]:
        """저장된 모든 공정 ID 목록 조회"""
        process_ids = []
        # 저장된 공정 데이터 파일에서
        if os.path.exists(self.process_data_dir):
            for filename in os.listdir(self.process_data_dir):
                if filename.endswith('.json'):
                    process_ids.append(filename.replace('.json', ''))
        # 구간 데이터에서도 공정 ID 추가
        for interval in self.intervals:
            if interval.get('process_id') and interval['process_id'] not in process_ids:
                process_ids.append(interval['process_id'])
        return sorted(process_ids, reverse=True)

    def delete_process(self, process_id: str) -> Dict:
        """특정 공정의 모든 데이터 삭제"""
        deleted = {
            'process_data': False,
            'intervals': 0,
            'logs': 0,
            'report': False
        }

        # 1. 공정 데이터 파일 삭제
        process_file = os.path.join(self.process_data_dir, f'{process_id}.json')
        if os.path.exists(process_file):
            os.remove(process_file)
            deleted['process_data'] = True
            print(f"[INFO] 공정 데이터 삭제: {process_file}")

        # 2. 리포트 파일 삭제
        report_file = os.path.join(self.log_dir, f'report_{process_id}.json')
        if os.path.exists(report_file):
            os.remove(report_file)
            deleted['report'] = True
            print(f"[INFO] 리포트 삭제: {report_file}")

        # 3. 구간 데이터에서 해당 공정 삭제
        original_interval_count = len(self.intervals)
        self.intervals = [i for i in self.intervals if i.get('process_id') != process_id]
        deleted['intervals'] = original_interval_count - len(self.intervals)
        if deleted['intervals'] > 0:
            self._save_intervals()
            print(f"[INFO] {deleted['intervals']}개 이상 구간 삭제")

        # 4. 로그에서 해당 공정 삭제
        original_log_count = len(self.logs)
        self.logs = [l for l in self.logs if l.get('process_id') != process_id]
        deleted['logs'] = original_log_count - len(self.logs)
        if deleted['logs'] > 0:
            self._save_logs()
            print(f"[INFO] {deleted['logs']}개 로그 삭제")

        return deleted

    def get_logs(self,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 anomaly_type: Optional[str] = None,
                 severity: Optional[str] = None,
                 process_id: Optional[str] = None) -> List[Dict]:
        """로그 검색"""
        filtered = self.logs

        if process_id:
            filtered = [l for l in filtered if l['process_id'] == process_id]
        if anomaly_type:
            filtered = [l for l in filtered if l['anomaly_type'] == anomaly_type]
        if severity:
            filtered = [l for l in filtered if l['severity'] == severity]
        if start_date:
            filtered = [l for l in filtered if l['created_at'] >= start_date]
        if end_date:
            filtered = [l for l in filtered if l['created_at'] <= end_date]

        return filtered

    def get_intervals(self,
                      anomaly_type: Optional[str] = None,
                      severity: Optional[str] = None,
                      process_id: Optional[str] = None) -> List[Dict]:
        """이상 구간 검색"""
        filtered = self.intervals

        if process_id:
            filtered = [i for i in filtered if i['process_id'] == process_id]
        if anomaly_type:
            filtered = [i for i in filtered if i['anomaly_type'] == anomaly_type]
        if severity:
            filtered = [i for i in filtered if i['severity'] == severity]

        return filtered

    def generate_report(self, process_id: str) -> Dict:
        """공정별 분석 리포트 생성 (구간 기반)"""
        process_intervals = self.get_intervals(process_id=process_id)

        if not process_intervals:
            return {
                'process_id': process_id,
                'total_intervals': 0,
                'total_anomalies': 0,
                'summary': '이상 없음'
            }

        report = {
            'process_id': process_id,
            'total_intervals': len(process_intervals),  # 이상 구간 수
            'total_anomalies': len(process_intervals),  # 표시용 (구간 = 이상 1회)
            'by_type': {},
            'by_severity': {'warning': 0, 'critical': 0},
            'intervals': [],  # 상세 구간 정보
            'recommendations': []
        }

        for interval in process_intervals:
            # 타입별 집계
            atype = interval['anomaly_type']
            if atype not in report['by_type']:
                report['by_type'][atype] = 0
            report['by_type'][atype] += 1

            # 심각도별 집계
            report['by_severity'][interval['severity']] += 1

            # 구간 상세 정보
            report['intervals'].append({
                'id': interval['id'],
                'anomaly_type': atype,
                'severity': interval['severity'],
                'start_timestamp': interval['start_timestamp'],
                'end_timestamp': interval['end_timestamp'],
                'duration_points': interval['duration_points'],
                'analysis': interval['analysis'],
                'peak_values': interval['peak_values'],
                'data_points': interval['data_points']
            })

        # 권장 조치 생성
        if report['by_type'].get('MAGNET_CONTAMINATION', 0) > 0:
            count = report['by_type']['MAGNET_CONTAMINATION']
            report['recommendations'].append({
                'issue': f'마그넷 오염 감지 ({count}회)',
                'action': '마그넷 상태 점검 및 클리닝 필요. 플라즈마 밀도 저하로 인한 증착 품질 저하 우려.',
                'priority': 'high' if count > 2 else 'medium'
            })
        if report['by_type'].get('GAS_FLOW_ANOMALY', 0) > 0:
            count = report['by_type']['GAS_FLOW_ANOMALY']
            report['recommendations'].append({
                'issue': f'가스 유량 이상 감지 ({count}회)',
                'action': 'MFC(Mass Flow Controller) 점검 필요. 가스 라인 누수 또는 밸브 고장 가능성 확인.',
                'priority': 'high' if count > 2 else 'medium'
            })
        if report['by_type'].get('AR_FLUCTUATION', 0) > 0:
            count = report['by_type']['AR_FLUCTUATION']
            report['recommendations'].append({
                'issue': f'Ar 유량 진동 감지 ({count}회)',
                'action': 'MFC 제어 안정성 점검 필요. 방전 불안정/overpressure로 인한 이온화 변동 우려.',
                'priority': 'high' if count > 2 else 'medium'
            })
        if report['by_type'].get('PWPDS_PREDICTION_ANOMALY', 0) > 0:
            count = report['by_type']['PWPDS_PREDICTION_ANOMALY']
            report['recommendations'].append({
                'issue': f'PWPDS 예측 이상 감지 ({count}회)',
                'action': 'CNN 예측값 대비 실측값 이상 감지. 마그넷 오염 또는 타겟-웨이퍼 거리 변화 확인 필요.',
                'priority': 'high' if count > 1 else 'medium'
            })

        return report
