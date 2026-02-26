#!/usr/bin/env python3
"""
실시간 PVD 공정 이상감지 모니터

실시간으로 생성되는 CSV 파일을 모니터링하고 이상을 감지합니다.
감지된 이상은 로그로 저장되고 웹 API를 통해 조회할 수 있습니다.
"""

import os
import sys
import time
import json
import glob
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Optional, Set
from pathlib import Path
import csv

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.anomaly_detector import PVDAnomalyDetector, AnomalyLogger, AnomalyResult, AnomalyInterval
from cnn_inference import get_cnn_engine, CNNInferenceEngine
from lstm_inference import get_inference_engine, PVD4InferenceEngine
from aging_inference import get_aging_engine, AgingInferenceEngine


class FileWatcher:
    """CSV 파일 변경 감지 클래스"""

    def __init__(self, watch_dir: str):
        self.watch_dir = watch_dir
        self.file_positions: Dict[str, int] = {}
        self.processed_files: Set[str] = set()

    def get_new_files(self) -> List[str]:
        """새로운 CSV 파일 목록 반환"""
        patterns = [
            os.path.join(self.watch_dir, 'PVD4_NEW_*.csv'),
            os.path.join(self.watch_dir, 'PVD4_[0-9][0-9][0-9][0-9][0-9][0-9]_*.csv')
        ]

        current_files = set()
        for pattern in patterns:
            current_files.update(glob.glob(pattern))

        new_files = current_files - self.processed_files
        return sorted(new_files)

    def get_new_lines(self, filepath: str) -> List[str]:
        """파일에서 새로운 라인 읽기"""
        if not os.path.exists(filepath):
            return []

        current_pos = self.file_positions.get(filepath, 0)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                f.seek(current_pos)
                new_lines = f.readlines()
                self.file_positions[filepath] = f.tell()

            # 빈 라인 제거
            return [line.strip() for line in new_lines if line.strip()]
        except Exception as e:
            print(f"[ERROR] 파일 읽기 실패: {filepath} - {e}")
            return []

    def mark_processed(self, filepath: str):
        """파일을 처리 완료로 표시"""
        self.processed_files.add(filepath)
        if filepath in self.file_positions:
            del self.file_positions[filepath]


class RealtimeMonitor:
    """실시간 이상감지 모니터"""

    def __init__(self,
                 watch_dir: str,
                 log_dir: str,
                 check_interval: float = 0.5,
                 callback=None):
        """
        Args:
            watch_dir: 모니터링할 CSV 디렉토리
            log_dir: 로그 저장 디렉토리
            check_interval: 파일 체크 간격 (초)
            callback: 이상 발견 시 호출할 콜백 함수
        """
        self.watch_dir = watch_dir
        self.log_dir = log_dir
        self.check_interval = check_interval
        self.callback = callback

        self.detector = PVDAnomalyDetector()
        self.logger = AnomalyLogger(log_dir)
        self.watcher = FileWatcher(watch_dir)

        self.running = False
        self.current_file: Optional[str] = None
        self.current_process_id: Optional[str] = None
        self.header: List[str] = []

        # CNN/LSTM 추론 엔진
        self.cnn_engine: CNNInferenceEngine = get_cnn_engine()
        self.lstm_engine: PVD4InferenceEngine = get_inference_engine()
        self.aging_engine: AgingInferenceEngine = get_aging_engine()

        # 실시간 데이터 버퍼 (전체 공정 데이터)
        self.data_buffer: List[Dict] = []
        self.max_buffer_size = 10000  # 공정 전체 데이터 저장

        # 현재 상태
        self.status = {
            'monitoring': False,
            'current_file': None,
            'current_process_id': None,
            'total_processed_lines': 0,
            'total_anomalies': 0,  # 이상 구간 수로 변경
            'total_anomaly_points': 0,  # 개별 이상 포인트 수
            'last_update': None,
            'active_anomalies': []  # 현재 진행 중인 이상 유형
        }

        # 이상 구간 정보 (실시간 차트 하이라이트용)
        self.anomaly_intervals: List[Dict] = []

        # 스레드 동기화
        self.lock = threading.Lock()

    def start(self):
        """모니터링 시작"""
        self.running = True
        self.status['monitoring'] = True
        print(f"[INFO] 모니터링 시작: {self.watch_dir}")

        while self.running:
            try:
                self._process_files()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] 모니터링 오류: {e}")
                time.sleep(1)

        self.status['monitoring'] = False
        print("[INFO] 모니터링 종료")

    def stop(self):
        """모니터링 중지"""
        self.running = False

    def _process_files(self):
        """파일 처리"""
        # 새 파일 확인
        new_files = self.watcher.get_new_files()
        if new_files and self.current_file is None:
            # 가장 최근 파일 선택
            self.current_file = new_files[-1]
            self._extract_process_id()
            self.detector.reset()
            self._init_engines_for_process()
            print(f"[INFO] 새 공정 감지: {self.current_process_id}")

        # 현재 파일 처리
        if self.current_file:
            self._process_current_file()

    def _init_engines_for_process(self):
        """새 공정 시작 시 추론 엔진 초기화"""
        # CNN 엔진 로드 및 초기화
        if not self.cnn_engine.loaded:
            self.cnn_engine.load_model()
        self.cnn_engine.reset()
        if self.current_process_id:
            self.cnn_engine.start_csv_export(self.current_process_id)

        # LSTM 엔진 로드 및 초기화
        if not self.lstm_engine.loaded:
            self.lstm_engine.load_model()
        self.lstm_engine.reset()

        # Aging 엔진 로드 및 초기화
        if not self.aging_engine.loaded:
            self.aging_engine.load_model()
        self.aging_engine.reset()

    def _extract_process_id(self):
        """파일명에서 공정 ID 추출"""
        if self.current_file:
            filename = os.path.basename(self.current_file)
            # PVD4_NEW_20250101_120000.csv -> 20250101_120000
            # PVD4_251015_20251015_153911.csv -> 20251015_153911
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                self.current_process_id = '_'.join(parts[-2:])
            else:
                self.current_process_id = filename.replace('.csv', '')

        self.status['current_file'] = self.current_file
        self.status['current_process_id'] = self.current_process_id

    def _process_current_file(self):
        """현재 파일의 새 라인 처리"""
        new_lines = self.watcher.get_new_lines(self.current_file)

        for line in new_lines:
            # 헤더 처리
            if not self.header:
                self.header = line.split(',')
                continue

            # 데이터 파싱
            values = line.split(',')
            if len(values) != len(self.header):
                continue

            data = dict(zip(self.header, values))

            # 수치형 변환
            for key in data:
                if key != 'Timer':
                    try:
                        data[key] = float(data[key])
                    except (ValueError, TypeError):
                        pass

            # 현재 데이터 인덱스 추가 (차트 하이라이트용)
            data['_index'] = self.status['total_processed_lines']

            # 타임스탬프 추출
            timestamp = data.get('Timer', datetime.now().isoformat())

            # CNN 추론 수행
            cnn_result = self.cnn_engine.predict(data, timestamp)
            if cnn_result:
                data['_cnn_predicted'] = cnn_result['predicted_pwpds']
                data['_cnn_error_pct'] = cnn_result['error_pct']
                data['_cnn_severity'] = cnn_result.get('severity', '')
                data['_cnn_cause'] = cnn_result['cause_info']['cause'] if cnn_result.get('cause_info') else ''

            # LSTM 추론 수행
            lstm_result = self.lstm_engine.predict(data, timestamp)
            if lstm_result:
                data['_lstm_predicted'] = lstm_result['predicted_pwpds']
                data['_lstm_error_pct'] = lstm_result['error_pct']

            # Aging (Extended LSTM) 추론 수행
            self.aging_engine.predict(data, timestamp)

            # 버퍼에 추가
            with self.lock:
                self.data_buffer.append(data)
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer.pop(0)

            # 이상 탐지 수행 (구간 기반)
            anomalies, new_intervals, ended_intervals = self.detector.detect(data)

            # 새 이상 구간 시작
            for interval in new_intervals:
                self._handle_new_interval(interval, data)

            # 이상 구간 종료
            for interval in ended_intervals:
                self._handle_ended_interval(interval)

            # 개별 이상 처리 (로그용)
            for anomaly in anomalies:
                self._handle_anomaly(anomaly, data)

            # 상태 업데이트
            self.status['total_processed_lines'] += 1
            self.status['total_anomalies'] = self.detector.get_interval_count()
            self.status['active_anomalies'] = list(self.detector.active_intervals.keys())
            self.status['last_update'] = datetime.now().isoformat()

        # 파일 종료 감지 (10초 이상 새 데이터 없음)
        if self.current_file and not new_lines:
            file_mtime = os.path.getmtime(self.current_file)
            if time.time() - file_mtime > 10:
                self._finish_process()

    def _handle_new_interval(self, interval: AnomalyInterval, data: Dict):
        """새 이상 구간 시작 처리"""
        interval_info = {
            'id': interval.interval_id,
            'type': interval.anomaly_type,
            'severity': interval.severity,
            'start_index': interval.start_index,
            'end_index': interval.start_index,
            'start_timestamp': interval.start_timestamp,
            'is_active': True
        }

        with self.lock:
            self.anomaly_intervals.append(interval_info)

        # 콜백 호출 (새 구간 시작 알림)
        if self.callback:
            self.callback(
                AnomalyResult(
                    timestamp=interval.start_timestamp,
                    anomaly_type=interval.anomaly_type,
                    severity=interval.severity,
                    description=f"[구간 시작] {interval.description}",
                    related_values=interval.peak_values
                ),
                data,
                {
                    'type': 'interval_start',
                    'interval_id': interval.interval_id,
                    'anomaly_type': interval.anomaly_type,
                    'severity': interval.severity,
                    'start_index': interval.start_index,
                    'timestamp': interval.start_timestamp
                }
            )

        severity_icon = '!' if interval.severity == 'critical' else '?'
        print(f"[{severity_icon}INTERVAL START] {interval.start_timestamp} - {interval.anomaly_type}")
        print(f"                {interval.description}")

    def _handle_ended_interval(self, interval: AnomalyInterval):
        """이상 구간 종료 처리"""
        # 구간 정보 업데이트
        with self.lock:
            for inv in self.anomaly_intervals:
                if inv['id'] == interval.interval_id:
                    inv['end_index'] = interval.end_index
                    inv['end_timestamp'] = interval.end_timestamp
                    inv['is_active'] = False
                    inv['duration_points'] = (interval.end_index or 0) - interval.start_index + 1
                    break

        # 구간 로그 저장
        self.logger.add_interval(interval, self.current_process_id or 'unknown')

        print(f"[INTERVAL END] {interval.anomaly_type} - 지속: {(interval.end_index or 0) - interval.start_index + 1}포인트")

    def _handle_anomaly(self, anomaly: AnomalyResult, data: Dict):
        """개별 이상 포인트 처리"""
        # 로그 저장 (개별 포인트)
        log_entry = self.logger.add_log(anomaly, self.current_process_id or 'unknown')

        self.status['total_anomaly_points'] += 1

        # 진행 중인 구간 정보 업데이트 (end_index)
        with self.lock:
            for inv in self.anomaly_intervals:
                if inv['type'] == anomaly.anomaly_type and inv['is_active']:
                    inv['end_index'] = data.get('_index', self.status['total_processed_lines'])

    def _finish_process(self):
        """공정 완료 처리"""
        print(f"[INFO] 공정 완료: {self.current_process_id}")

        # CNN CSV export 마무리
        self.cnn_engine.finish_csv_export()

        # 진행 중인 이상 구간 종료
        for atype, interval in list(self.detector.active_intervals.items()):
            interval.is_active = False
            interval.end_timestamp = datetime.now().strftime('[%Y.%m.%d %H:%M:%S]')
            interval.end_index = self.status['total_processed_lines'] - 1
            self.logger.add_interval(interval, self.current_process_id or 'unknown')
            print(f"[INTERVAL END] {atype} - 공정 종료로 구간 마감")

        # 공정 데이터 저장 (이력 조회용) - 예측값 포함
        if self.current_process_id:
            with self.lock:
                # CNN/LSTM 예측 데이터도 포함
                cnn_chart = self.cnn_engine.get_chart_data(10000)
                lstm_chart = self.lstm_engine.get_chart_data(10000)
                aging_chart = self.aging_engine.get_chart_data(10000)
                process_data = {
                    'process_id': self.current_process_id,
                    'data_points': self.data_buffer.copy(),
                    'intervals': self.anomaly_intervals.copy(),
                    'total_points': len(self.data_buffer),
                    'completed_at': datetime.now().isoformat(),
                    'cnn_predictions': cnn_chart.get('predictions', []),
                    'cnn_actuals': cnn_chart.get('actuals', []),
                    'lstm_predictions': lstm_chart.get('predictions', []),
                    'lstm_actuals': lstm_chart.get('actuals', []),
                    'cnn_anomaly_logs': self.cnn_engine.get_anomaly_logs(100),
                    'lstm_anomaly_logs': self.lstm_engine.get_anomaly_logs(100),
                    'aging_chart': aging_chart,
                    'aging_anomaly_logs': self.aging_engine.get_anomaly_logs(100),
                }
            self.logger.save_process_data(self.current_process_id, process_data)

        # 리포트 생성
        if self.current_process_id:
            report = self.logger.generate_report(self.current_process_id)
            report_path = os.path.join(
                self.log_dir,
                f'report_{self.current_process_id}.json'
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 리포트 저장: {report_path}")

        # 상태 초기화
        self.watcher.mark_processed(self.current_file)
        self.current_file = None
        self.current_process_id = None
        self.header = []
        self.detector.reset()

        with self.lock:
            self.data_buffer.clear()
            self.anomaly_intervals.clear()

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return self.status.copy()

    def get_recent_data(self, count: int = 50) -> List[Dict]:
        """최근 데이터 반환"""
        with self.lock:
            return self.data_buffer[-count:]

    def get_logs(self, **filters) -> List[Dict]:
        """로그 조회"""
        return self.logger.get_logs(**filters)

    def get_report(self, process_id: str) -> Dict:
        """공정 리포트 조회"""
        return self.logger.generate_report(process_id)

    def get_anomaly_intervals(self) -> List[Dict]:
        """현재 공정의 이상 구간 목록 반환 (차트 하이라이트용)"""
        with self.lock:
            return self.anomaly_intervals.copy()

    def get_intervals(self, process_id: Optional[str] = None) -> List[Dict]:
        """저장된 이상 구간 조회"""
        return self.logger.get_intervals(process_id=process_id)

    def get_process_data(self, process_id: str) -> Optional[Dict]:
        """저장된 공정 데이터 조회"""
        return self.logger.get_process_data(process_id)

    def get_all_process_ids(self) -> List[str]:
        """모든 공정 ID 목록 조회"""
        return self.logger.get_all_process_ids()

    def delete_process(self, process_id: str) -> Dict:
        """공정 데이터 삭제"""
        return self.logger.delete_process(process_id)


# 전역 모니터 인스턴스 (웹 API에서 사용)
_monitor_instance: Optional[RealtimeMonitor] = None


def get_monitor() -> Optional[RealtimeMonitor]:
    """전역 모니터 인스턴스 반환"""
    return _monitor_instance


def set_monitor(monitor: RealtimeMonitor):
    """전역 모니터 인스턴스 설정"""
    global _monitor_instance
    _monitor_instance = monitor


def main():
    parser = argparse.ArgumentParser(description='PVD 공정 실시간 이상감지 모니터')
    parser.add_argument('--watch-dir',
                        default='/home/goo4168/baco/pvd_monitoring/data',
                        help='모니터링할 디렉토리')
    parser.add_argument('--log-dir',
                        default='/home/goo4168/baco/pvd_monitoring/logs',
                        help='로그 저장 디렉토리')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='체크 간격 (초)')

    args = parser.parse_args()

    # 디렉토리 생성
    os.makedirs(args.watch_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 모니터 생성 및 시작
    monitor = RealtimeMonitor(
        watch_dir=args.watch_dir,
        log_dir=args.log_dir,
        check_interval=args.interval
    )
    set_monitor(monitor)

    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n[INFO] 모니터링 중단됨")
        sys.exit(0)


if __name__ == '__main__':
    main()
