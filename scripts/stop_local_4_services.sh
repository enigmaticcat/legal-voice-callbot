#!/usr/bin/env bash
set -euo pipefail

# ── Auto-detect PID file ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$SCRIPT_DIR/../asr" ]]; then
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -d "$SCRIPT_DIR/../nutrition-callbot/asr" ]]; then
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

PID_FILE="$ROOT_DIR/.local_services.pids"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[INFO] Không tìm thấy PID file: $PID_FILE"
  exit 0
fi

# ── Đọc PID ────────────────────────────────────────────────────────────
read_pid() {
  local key="$1"
  local value
  value=$(grep -E "^${key}=" "$PID_FILE" | tail -n 1 | cut -d'=' -f2- || true)
  echo "$value" | tr -cd '0-9'
}

ASR=$(read_pid "ASR")
BRAIN=$(read_pid "BRAIN")
TTS=$(read_pid "TTS")
GATEWAY=$(read_pid "GATEWAY")
WEB=$(read_pid "WEB")

# ── Stop từng service ──────────────────────────────────────────────────
stop_one() {
  local name="$1"
  local pid="$2"

  if [[ -z "${pid:-}" ]]; then
    echo "[SKIP] $name — PID trống"
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "[STOP] $name (PID $pid)"
  else
    echo "[SKIP] $name — PID $pid không còn chạy"
  fi
}

stop_one "web"     "${WEB:-}"
stop_one "gateway" "${GATEWAY:-}"
stop_one "tts"     "${TTS:-}"
stop_one "brain"   "${BRAIN:-}"
stop_one "asr"     "${ASR:-}"

# Dọn vite/node còn sót trên WEB_PORT
lsof -ti tcp:"${WEB_PORT:-3000}" 2>/dev/null | xargs kill 2>/dev/null || true

rm -f "$PID_FILE"
echo "[OK] Đã dừng services và xóa PID file."
