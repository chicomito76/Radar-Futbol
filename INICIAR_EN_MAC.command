#!/bin/zsh
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 no está instalado."
  echo "Instala Python 3 y vuelve a abrir este archivo."
  read -k 1
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r backend/requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "Se creó el archivo .env."
  echo "Agrega API_SPORTS_KEY para datos reales."
fi

set -a
source .env
set +a

(open "http://127.0.0.1:8787" >/dev/null 2>&1 &) || true
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8787
