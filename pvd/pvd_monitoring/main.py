#!/usr/bin/env python3
"""
PVD 공정 실시간 이상감지 웹 플랫폼

FastAPI 기반 웹 애플리케이션으로 실시간 데이터 시각화와 이상 로그 관리 기능 제공
"""

import os
import sys
import json
import asyncio
import threading
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.requests import Request
from pydantic import BaseModel

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abnormal_monitor import RealtimeMonitor, set_monitor, get_monitor
from models.anomaly_detector import AnomalyLogger, AnomalyResult
from lstm_inference import get_inference_engine, PVD4InferenceEngine
from cnn_inference import get_cnn_engine, CNNInferenceEngine
from aging_inference import get_aging_engine

# 설정
WATCH_DIR = os.environ.get('PVD_WATCH_DIR', '/home/goo4168/baco/pvd_monitoring/data')
LOG_DIR = os.environ.get('PVD_LOG_DIR', '/home/goo4168/baco/pvd_monitoring/logs')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
EXPORT_DIR = os.environ.get('PVD_EXPORT_DIR', '/home/goo4168/baco/pvd_monitoring/exports')

# 디렉토리 생성
os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# 이상 발생 시 WebSocket으로 브로드캐스트
def anomaly_callback(anomaly: AnomalyResult, data: Dict, log_entry: Dict):
    """이상 발생 시 콜백"""
    message = {
        'type': 'anomaly',
        'data': log_entry
    }
    # 비동기 브로드캐스트를 위한 이벤트 루프 처리
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(manager.broadcast(message))
    except RuntimeError:
        pass

# 모니터 스레드
monitor_thread: Optional[threading.Thread] = None

# 데이터 생성기 프로세스
generator_process: Optional[subprocess.Popen] = None
generator_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 처리"""
    global monitor_thread

    # 모니터 시작
    monitor = RealtimeMonitor(
        watch_dir=WATCH_DIR,
        log_dir=LOG_DIR,
        check_interval=0.5,
        callback=anomaly_callback
    )
    set_monitor(monitor)

    monitor_thread = threading.Thread(target=monitor.start, daemon=True)
    monitor_thread.start()
    print(f"[INFO] 모니터링 시작됨: {WATCH_DIR}")

    yield

    # 모니터 종료
    monitor.stop()
    print("[INFO] 모니터링 종료됨")

# FastAPI 앱 생성
app = FastAPI(
    title="PVD 공정 이상감지 플랫폼",
    description="반도체 PVD 증착 장비 센서 데이터 실시간 이상감지 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# 정적 파일 및 템플릿
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ============== API 모델 ==============

class LogSearchParams(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    anomaly_type: Optional[str] = None
    severity: Optional[str] = None
    process_id: Optional[str] = None


# ============== 웹 페이지 ==============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})


# ============== REST API ==============

@app.get("/api/status")
async def get_status():
    """모니터링 상태 조회"""
    monitor = get_monitor()
    if monitor:
        return monitor.get_status()
    return {"monitoring": False, "error": "Monitor not initialized"}


@app.get("/api/data/realtime")
async def get_realtime_data(count: int = Query(default=10000, le=10000)):
    """실시간 데이터 조회 (전체 공정 데이터)"""
    monitor = get_monitor()
    if monitor:
        return {"data": monitor.get_recent_data(count)}
    return {"data": []}


@app.get("/api/logs")
async def get_logs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    severity: Optional[str] = None,
    process_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000)
):
    """이상 로그 조회"""
    monitor = get_monitor()
    if monitor:
        logs = monitor.get_logs(
            start_date=start_date,
            end_date=end_date,
            anomaly_type=anomaly_type,
            severity=severity,
            process_id=process_id
        )
        return {"logs": logs[-limit:], "total": len(logs)}
    return {"logs": [], "total": 0}


@app.get("/api/logs/summary")
async def get_logs_summary():
    """로그 요약 통계"""
    monitor = get_monitor()
    if not monitor:
        return {"error": "Monitor not initialized"}

    logs = monitor.get_logs()

    summary = {
        "total": len(logs),
        "by_type": {},
        "by_severity": {"warning": 0, "critical": 0},
        "recent_24h": 0
    }

    now = datetime.now()
    for log in logs:
        # 타입별
        atype = log['anomaly_type']
        summary['by_type'][atype] = summary['by_type'].get(atype, 0) + 1

        # 심각도별
        summary['by_severity'][log['severity']] += 1

        # 최근 24시간
        try:
            log_time = datetime.fromisoformat(log['created_at'])
            if (now - log_time).total_seconds() < 86400:
                summary['recent_24h'] += 1
        except:
            pass

    return summary


@app.get("/api/report/{process_id}")
async def get_report(process_id: str):
    """공정별 분석 리포트 조회"""
    monitor = get_monitor()
    if monitor:
        return monitor.get_report(process_id)
    return {"error": "Monitor not initialized"}


@app.get("/api/processes")
async def get_processes():
    """처리된 공정 목록"""
    monitor = get_monitor()
    if not monitor:
        return {"processes": []}

    intervals = monitor.get_intervals()
    process_ids = list(set(inv['process_id'] for inv in intervals))
    # 로그에서도 공정 ID 추가 (하위 호환)
    logs = monitor.get_logs()
    process_ids.extend([log['process_id'] for log in logs])
    process_ids = list(set(process_ids))
    process_ids.sort(reverse=True)

    return {"processes": process_ids}


@app.get("/api/intervals")
async def get_intervals(process_id: Optional[str] = None):
    """이상 구간 조회"""
    monitor = get_monitor()
    if not monitor:
        return {"intervals": []}

    intervals = monitor.get_intervals(process_id=process_id)
    return {"intervals": intervals, "total": len(intervals)}


@app.get("/api/intervals/realtime")
async def get_realtime_intervals():
    """실시간 이상 구간 조회 (차트 하이라이트용)"""
    monitor = get_monitor()
    if not monitor:
        return {"intervals": []}

    intervals = monitor.get_anomaly_intervals()
    return {"intervals": intervals}


@app.get("/api/history/processes")
async def get_history_processes():
    """저장된 모든 공정 목록 조회"""
    monitor = get_monitor()
    if not monitor:
        return {"processes": []}

    process_ids = monitor.get_all_process_ids()
    return {"processes": process_ids}


@app.get("/api/history/{process_id}")
async def get_process_history(process_id: str):
    """특정 공정의 전체 데이터 조회 (차트용 데이터 + 이상 구간)"""
    monitor = get_monitor()
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not initialized")

    # 저장된 공정 데이터 조회
    process_data = monitor.get_process_data(process_id)

    # 해당 공정의 이상 구간 조회
    intervals = monitor.get_intervals(process_id=process_id)

    # 리포트도 함께 조회
    report = monitor.get_report(process_id)

    if not process_data and not intervals:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")

    return {
        "process_id": process_id,
        "data_points": process_data.get('data_points', []) if process_data else [],
        "intervals": process_data.get('intervals', intervals) if process_data else intervals,
        "total_points": process_data.get('total_points', 0) if process_data else 0,
        "completed_at": process_data.get('completed_at', '') if process_data else '',
        "report": report
    }


# ============== Column Ranges & Export API ==============

@app.get("/api/column-ranges")
async def get_column_ranges():
    """차트 Y축 범위용 column_ranges 반환"""
    engine = get_cnn_engine()
    if not engine.loaded:
        engine.load_model()
    return engine.get_column_ranges()


@app.get("/api/exports")
async def list_exports():
    """CSV export 파일 목록"""
    files = []
    if os.path.exists(EXPORT_DIR):
        for f in sorted(os.listdir(EXPORT_DIR), reverse=True):
            if f.endswith('.csv'):
                fpath = os.path.join(EXPORT_DIR, f)
                stat = os.stat(fpath)
                files.append({
                    'filename': f,
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
    return {"exports": files}


@app.get("/api/exports/{filename}")
async def download_export(filename: str):
    """CSV 파일 다운로드"""
    filepath = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    # 경로 traversal 방지
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return FileResponse(filepath, media_type='text/csv', filename=filename)


# ============== CNN 추론 API ==============

@app.get("/api/cnn/status")
async def get_cnn_status():
    """CNN 추론 엔진 상태 조회"""
    engine = get_cnn_engine()
    return engine.get_status()


@app.get("/api/cnn/chart-data")
async def get_cnn_chart_data(count: int = Query(default=200, le=1000)):
    """CNN 추론 차트 데이터 조회"""
    engine = get_cnn_engine()
    return engine.get_chart_data(count)


@app.get("/api/cnn/anomaly-logs")
async def get_cnn_anomaly_logs(limit: int = Query(default=100, le=500)):
    """CNN 추론 이상 로그 조회"""
    engine = get_cnn_engine()
    return {"logs": engine.get_anomaly_logs(limit)}


# ============== 데이터 생성기 제어 API ==============

@app.get("/api/generator/status")
async def get_generator_status():
    """데이터 생성기 상태 조회"""
    global generator_process
    with generator_lock:
        if generator_process is None:
            return {"running": False}
        # 프로세스가 아직 실행 중인지 확인
        poll = generator_process.poll()
        if poll is None:
            return {"running": True, "pid": generator_process.pid}
        else:
            generator_process = None
            return {"running": False}


@app.post("/api/generator/start")
async def start_generator(mode: str = "abnormal"):
    """데이터 생성기 시작"""
    global generator_process

    with generator_lock:
        # 이미 실행 중인지 확인
        if generator_process is not None:
            poll = generator_process.poll()
            if poll is None:
                return {"success": False, "message": "이미 실행 중입니다", "running": True, "pid": generator_process.pid}
            else:
                generator_process = None

        # 새 프로세스 시작
        try:
            project_dir = os.path.dirname(os.path.abspath(__file__))

            if mode == "normal":
                # 정상 공정: 1공정만, 이상 주입 없음
                cmd = ["python3", "generate_data.py"]
            else:
                # 비정상 공정: 1공정만, PWPDS.Data 이상 주입
                cmd = ["python3", "generate_data.py", "--inject-anomaly"]

            generator_process = subprocess.Popen(
                cmd,
                cwd=project_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            mode_text = "정상 공정" if mode == "normal" else "비정상 공정 (이상 주입)"
            return {"success": True, "message": f"데이터 생성기가 시작되었습니다 ({mode_text})", "running": True, "pid": generator_process.pid, "mode": mode}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"시작 실패: {str(e)}", "running": False}


@app.delete("/api/history/{process_id}")
async def delete_process_history(process_id: str):
    """특정 공정의 모든 데이터 삭제"""
    monitor = get_monitor()
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not initialized")

    try:
        deleted_items = monitor.delete_process(process_id)
        return {
            "success": True,
            "message": f"공정 {process_id}의 데이터가 삭제되었습니다",
            "deleted": deleted_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 실패: {str(e)}")


@app.post("/api/generator/stop")
async def stop_generator():
    """데이터 생성기 중지"""
    global generator_process
    with generator_lock:
        if generator_process is None:
            return {"success": False, "message": "실행 중인 생성기가 없습니다", "running": False}

        poll = generator_process.poll()
        if poll is not None:
            generator_process = None
            return {"success": False, "message": "이미 종료되었습니다", "running": False}

        try:
            generator_process.terminate()
            generator_process.wait(timeout=5)
            generator_process = None
            return {"success": True, "message": "데이터 생성기가 중지되었습니다", "running": False}
        except Exception as e:
            return {"success": False, "message": f"중지 실패: {str(e)}", "running": True}


# ============== WebSocket ==============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 데이터 스트리밍"""
    await manager.connect(websocket)
    try:
        monitor = get_monitor()
        while True:
            # 상태 및 최신 데이터 전송
            if monitor:
                status = monitor.get_status()
                # 전체 공정 데이터 전송 (최대 10000개)
                all_data = monitor.get_recent_data(10000)
                anomaly_intervals = monitor.get_anomaly_intervals()

                await websocket.send_json({
                    'type': 'update',
                    'status': status,
                    'data': all_data,
                    'intervals': anomaly_intervals
                })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============== LSTM 추론 API ==============

@app.get("/api/inference/status")
async def get_inference_status():
    """LSTM 추론 엔진 상태 조회"""
    engine = get_inference_engine()
    return engine.get_status()


@app.post("/api/inference/load")
async def load_inference_model():
    """LSTM 모델 로드"""
    engine = get_inference_engine()
    success = engine.load_model()
    return {"success": success, "status": engine.get_status()}


@app.post("/api/inference/reset")
async def reset_inference():
    """추론 버퍼 초기화"""
    engine = get_inference_engine()
    engine.reset()
    return {"success": True, "message": "추론 버퍼가 초기화되었습니다"}


@app.get("/api/inference/chart-data")
async def get_inference_chart_data(count: int = Query(default=100, le=1000)):
    """추론 차트 데이터 조회"""
    engine = get_inference_engine()
    return engine.get_chart_data(count)


@app.get("/api/inference/anomaly-logs")
async def get_inference_anomaly_logs(limit: int = Query(default=100, le=500)):
    """추론 이상 로그 조회"""
    engine = get_inference_engine()
    return {"logs": engine.get_anomaly_logs(limit)}


# WebSocket 연결 관리 (추론용)
class InferenceConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

inference_manager = InferenceConnectionManager()


@app.websocket("/ws/inference")
async def websocket_inference_endpoint(websocket: WebSocket):
    """실시간 CNN 추론 WebSocket (실시간 모니터링 탭)"""
    await inference_manager.connect(websocket)
    cnn_engine = get_cnn_engine()

    # 모델 로드
    if not cnn_engine.loaded:
        cnn_engine.load_model()

    try:
        monitor = get_monitor()
        last_processed_lines = 0

        while True:
            if monitor and cnn_engine.loaded:
                current_lines = monitor.status.get('total_processed_lines', 0)
                if current_lines > last_processed_lines:
                    last_processed_lines = current_lines

                    # CNN 차트 데이터 (abnormal_monitor에서 이미 추론 완료)
                    chart_data = cnn_engine.get_chart_data(200)
                    anomaly_logs = cnn_engine.get_anomaly_logs(50)

                    # Ar 변동 정보 (이상 구간에서 AR_FLUCTUATION 추출)
                    ar_intervals = []
                    if monitor:
                        all_intervals = monitor.get_anomaly_intervals()
                        ar_intervals = [inv for inv in all_intervals if inv.get('type') == 'AR_FLUCTUATION']

                    await websocket.send_json({
                        'type': 'cnn_update',
                        'chart_data': chart_data,
                        'anomaly_logs': anomaly_logs,
                        'ar_intervals': ar_intervals,
                        'status': cnn_engine.get_status()
                    })

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        inference_manager.disconnect(websocket)


# WebSocket 연결 관리 (장비 노후 감지용)
aging_manager = ConnectionManager()


@app.websocket("/ws/aging")
async def websocket_aging_endpoint(websocket: WebSocket):
    """장비 노후 감지 WebSocket (Extended LSTM 기반, 7개 칼럼 5초 후 예측)"""
    await aging_manager.connect(websocket)
    aging_engine = get_aging_engine()

    if not aging_engine.loaded:
        aging_engine.load_model()

    try:
        monitor = get_monitor()
        last_processed_lines = 0

        while True:
            if monitor and aging_engine.loaded:
                current_lines = monitor.status.get('total_processed_lines', 0)
                if current_lines > last_processed_lines:
                    last_processed_lines = current_lines

                    chart_data = aging_engine.get_chart_data(200)
                    anomaly_logs = aging_engine.get_anomaly_logs(50)

                    await websocket.send_json({
                        'type': 'aging_update',
                        'chart_data': chart_data,
                        'anomaly_logs': anomaly_logs,
                        'status': aging_engine.get_status()
                    })

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        aging_manager.disconnect(websocket)


# ============== 직접 실행 ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
