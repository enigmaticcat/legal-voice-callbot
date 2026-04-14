#!/usr/bin/env bash
set -euo pipefail

# ── Auto-detect paths ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tìm STACK_DIR (thư mục chứa asr/ brain/ gateway/ tts/ web/)
if [[ -d "$SCRIPT_DIR/../asr" ]]; then
  # scripts/ nằm bên trong repo (Lightning / Colab / sau git clone)
  STACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -d "$SCRIPT_DIR/../nutrition-callbot/asr" ]]; then
  # scripts/ nằm ngoài repo, cạnh nutrition-callbot/ (local dev)
  STACK_DIR="$(cd "$SCRIPT_DIR/../nutrition-callbot" && pwd)"
else
  echo "[ERROR] Không tìm thấy project root từ $SCRIPT_DIR"
  echo "        Đảm bảo scripts/ nằm bên trong hoặc cạnh thư mục chứa asr/ gateway/ ..."
  exit 1
fi

ROOT_DIR="$(cd "$STACK_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/local-services"
PID_FILE="$ROOT_DIR/.local_services.pids"
ENV_FILE="$STACK_DIR/.env"

mkdir -p "$LOG_DIR"

echo "[INFO] STACK_DIR = $STACK_DIR"
echo "[INFO] ROOT_DIR  = $ROOT_DIR"

# ── Kiểm tra .env ──────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] Thiếu file env: $ENV_FILE"
  exit 1
fi

# ── Kích hoạt venv (tìm trong STACK_DIR rồi ROOT_DIR) ─────────────────
if [[ -f "$STACK_DIR/venv/bin/activate" ]]; then
  source "$STACK_DIR/venv/bin/activate"
  echo "[INFO] venv: $STACK_DIR/venv"
elif [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  source "$ROOT_DIR/venv/bin/activate"
  echo "[INFO] venv: $ROOT_DIR/venv"
else
  echo "[WARN] Không tìm thấy venv — dùng Python hệ thống"
fi

# ── Load .env ──────────────────────────────────────────────────────────
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[WARN] GEMINI_API_KEY trống — Brain cần key này nếu dùng Gemini backend"
fi

# ── Kiểm tra PID file ──────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
  echo "[WARN] PID file đã tồn tại: $PID_FILE"
  echo "       Chạy scripts/stop_local_4_services.sh trước."
  exit 1
fi

# ── Giải phóng port nếu bận ────────────────────────────────────────────
free_port_if_busy() {
  local port="$1"
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "[WARN] Port $port bận. Killing: $pids"
    for pid in $pids; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}

free_port_if_busy "${ASR_PORT:-50051}"
free_port_if_busy "${BRAIN_PORT:-50052}"
free_port_if_busy "${TTS_PORT:-50053}"
free_port_if_busy "${GATEWAY_PORT:-8000}"
free_port_if_busy "${WEB_PORT:-3000}"

# ── Helper: start background service ───────────────────────────────────
start_service() {
  local name="$1"
  local workdir="$2"
  local cmd="$3"
  local logfile="$LOG_DIR/${name}.log"

  echo "[START] $name → log: $logfile" >&2
  (
    cd "$workdir"
    nohup bash -c "$cmd" >"$logfile" 2>&1 &
    echo "$!"
  )
}

# ── Khởi động ASR / Brain / TTS ────────────────────────────────────────
ASR_PID=$(start_service "asr"     "$STACK_DIR/asr"     "python server.py")
BRAIN_PID=$(start_service "brain" "$STACK_DIR"         "python -m brain.server")
TTS_PID=$(start_service "tts"     "$STACK_DIR/tts"     "python server.py")

# ── Khởi động Gateway ──────────────────────────────────────────────────
GATEWAY_PID=$(start_service "gateway" "$STACK_DIR/gateway" \
  "uvicorn main:app --host 0.0.0.0 --port ${GATEWAY_PORT:-8000}")

# ── Web frontend ───────────────────────────────────────────────────────
# Nếu web/dist đã build → gateway sẽ serve static files, không cần Vite
if [[ -d "$STACK_DIR/web/dist" ]]; then
  echo "[SKIP] web — dist đã có, gateway sẽ serve static files"
  WEB_PID=""
else
  echo "[START] web (Vite dev server)" >&2
  WEB_PID=$(
    cd "$STACK_DIR/web"
    VITE_GATEWAY_URL="http://localhost:${GATEWAY_PORT:-8000}" \
      nohup npm run dev >"$LOG_DIR/web.log" 2>&1 &
    echo "$!"
  )
fi

# ── Lưu PID file ───────────────────────────────────────────────────────
cat >"$PID_FILE" <<EOF
ASR=$ASR_PID
BRAIN=$BRAIN_PID
TTS=$TTS_PID
GATEWAY=$GATEWAY_PID
WEB=${WEB_PID:-}
EOF

echo ""
echo "[OK] Services đã khởi động."
echo "     PID file : $PID_FILE"
echo "     Logs     : $LOG_DIR"
echo ""
echo "     Health check:"
echo "       curl http://localhost:${ASR_PORT:-50051}/health"
echo "       curl http://localhost:${BRAIN_PORT:-50052}/health"
echo "       curl http://localhost:${TTS_PORT:-50053}/health"
echo "       curl http://localhost:${GATEWAY_PORT:-8000}/health"
if [[ -d "$STACK_DIR/web/dist" ]]; then
  echo ""
  echo "     Frontend: http://localhost:${GATEWAY_PORT:-8000}"
else
  echo "     Frontend: http://localhost:${WEB_PORT:-3000}"
fi
