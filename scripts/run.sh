#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies ..."
pip install -q -r requirements.txt

if [[ ! -f diy.env ]] && [[ -f diy.env.example ]]; then
  cp diy.env.example diy.env
  echo "Created diy.env — edit OLLAMA_BASE_URL before use."
fi

export PYTHONPATH="${ROOT}"

# Load DIY_PORT safely without sourcing the whole file (values may contain quotes)
if [[ -f diy.env ]]; then
  DIY_PORT="$(grep -E '^DIY_PORT=' diy.env | head -1 | cut -d= -f2- | tr -d '\r"' | xargs || true)"
  export DIY_PORT
fi

pick_port() {
  local start="${1:-8780}"
  local port="$start"
  while [[ "$port" -lt $((start + 50)) ]]; do
    if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
      if ! lsof -i ":${port}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "$port"
        return 0
      fi
    fi
    port=$((port + 1))
  done
  echo "$start"
}

PREFERRED="${DIY_PORT:-8780}"
PORT="$(pick_port "$PREFERRED")"
if [[ "$PORT" != "$PREFERRED" ]]; then
  echo "Note: port ${PREFERRED} is in use — using ${PORT} instead."
  echo "      Stop the old server with: pkill -f 'uvicorn src.api.main:app'"
  echo ""
fi
HOST="${DIY_HOST:-0.0.0.0}"

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LAN_IP="${LAN_IP:-your-local-ip}"

echo ""
echo "  DIY AI — share on your network:"
echo "    http://${LAN_IP}:${PORT}"
echo "  On this machine:"
echo "    http://127.0.0.1:${PORT}"
echo "  Press Ctrl+C to stop."
echo ""

exec python -m uvicorn src.api.main:app --host "$HOST" --port "$PORT" --reload
