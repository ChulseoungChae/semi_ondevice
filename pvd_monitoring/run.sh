#!/bin/bash
#
# PVD 공정 이상감지 플랫폼 실행 스크립트
#

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 가상환경 확인
if [ -d "venv" ]; then
    source venv/bin/activate
fi

case "$1" in
    server)
        # 웹 서버만 실행 (모니터링 포함)
        echo "[INFO] 웹 서버 시작 (http://0.0.0.0:9300)"
        python3 -c "import uvicorn; from main import app; uvicorn.run(app, host='0.0.0.0', port=9300)"
        ;;

    generate)
        # 데이터 생성기 실행
        shift
        echo "[INFO] 데이터 생성기 시작"
        python3 generate_data.py "$@"
        ;;

    monitor)
        # 모니터링만 실행 (CLI)
        shift
        echo "[INFO] 이상감지 모니터 시작"
        python3 abnormal_monitor.py "$@"
        ;;

    demo)
        # 데모 모드: 데이터 생성 + 웹 서버 동시 실행
        echo "[INFO] 데모 모드 시작"
        echo "[INFO] 데이터 생성기를 백그라운드로 실행합니다..."
        python3 generate_data.py --continuous --inject-anomaly &
        GENERATOR_PID=$!
        echo "[INFO] 생성기 PID: $GENERATOR_PID"

        echo "[INFO] 웹 서버 시작 (http://0.0.0.0:9300)"
        trap "kill $GENERATOR_PID 2>/dev/null" EXIT
        python3 -c "import uvicorn; from main import app; uvicorn.run(app, host='0.0.0.0', port=9300)"
        ;;

    install)
        # 의존성 설치
        echo "[INFO] 의존성 설치 중..."
        pip3 install -r requirements.txt
        echo "[INFO] 설치 완료"
        ;;

    *)
        echo "사용법: $0 {server|generate|monitor|demo|install}"
        echo ""
        echo "명령어:"
        echo "  server   - 웹 서버 실행 (모니터링 포함)"
        echo "  generate - 테스트 데이터 생성"
        echo "  monitor  - CLI 모드로 이상감지 모니터 실행"
        echo "  demo     - 데모 모드 (데이터 생성 + 웹 서버)"
        echo "  install  - 의존성 설치"
        echo ""
        echo "예시:"
        echo "  $0 install                           # 의존성 설치"
        echo "  $0 server                            # 웹 서버 시작"
        echo "  $0 generate --continuous             # 연속 데이터 생성"
        echo "  $0 generate --inject-anomaly         # 이상 데이터 포함 생성"
        echo "  $0 demo                              # 데모 모드"
        exit 1
        ;;
esac
