#!/usr/bin/env bash
# Установка RAG Agent на чистом Ubuntu 24.04 LTS DigitalOcean Droplet.
# Запускать от root или через sudo:  sudo bash deploy/install.sh

set -euo pipefail

APP_USER="${APP_USER:-rag}"
APP_DIR="${APP_DIR:-/opt/rag_agent_v1}"
DATA_DIR="${DATA_DIR:-/data}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
DOMAIN="${DOMAIN:-}"   # для Caddy; если пусто — пропустим caddy

if [[ $EUID -ne 0 ]]; then
    echo "Запускайте через sudo." >&2
    exit 1
fi

echo "==> Обновляю apt и ставлю системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    software-properties-common \
    curl ca-certificates gnupg \
    git build-essential pkg-config \
    libmagic1 \
    poppler-utils \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    libreoffice --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    debian-keyring debian-archive-keyring apt-transport-https

echo "==> Создаю пользователя $APP_USER, если его нет"
if ! id "$APP_USER" >/dev/null 2>&1; then
    useradd -m -s /bin/bash "$APP_USER"
fi

echo "==> Готовлю каталог приложения $APP_DIR"
mkdir -p "$APP_DIR"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "==> Готовлю каталог данных $DATA_DIR"
mkdir -p "$DATA_DIR"
chown -R "$APP_USER:$APP_USER" "$DATA_DIR"

echo "==> Установка приложения (если репозитория ещё нет в $APP_DIR — клонируйте вручную)"
if [[ ! -f "$APP_DIR/requirements.txt" ]]; then
    echo "В $APP_DIR нет requirements.txt. Сначала склонируйте репозиторий:"
    echo "  sudo -u $APP_USER git clone https://github.com/Erofaxxx/rag_agent_v1 $APP_DIR"
    exit 1
fi

echo "==> Создаю виртуальное окружение и ставлю зависимости"
sudo -u "$APP_USER" $PYTHON_BIN -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip wheel
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$APP_DIR/.env" ]]; then
    echo "==> Копирую .env.example -> .env (отредактируйте перед стартом!)"
    sudo -u "$APP_USER" cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    sed -i "s|^DATA_DIR=.*|DATA_DIR=$DATA_DIR|" "$APP_DIR/.env"
fi

echo "==> Прогружаю BGE-M3 в кэш ~/.cache/huggingface (один раз ~2.3GB)"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/python" - <<'PY'
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
try:
    from FlagEmbedding import BGEM3FlagModel
    BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("OK: BGE-M3 загружен")
except Exception as e:
    print("WARN:", e)
PY

echo "==> Устанавливаю systemd-юнит"
SERVICE_FILE=/etc/systemd/system/rag-agent.service
cp "$APP_DIR/deploy/rag-agent.service" "$SERVICE_FILE"
sed -i "s|{{APP_USER}}|$APP_USER|g" "$SERVICE_FILE"
sed -i "s|{{APP_DIR}}|$APP_DIR|g" "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable rag-agent.service

if [[ -n "$DOMAIN" ]]; then
    echo "==> Ставлю Caddy и настраиваю реверс-прокси для $DOMAIN"
    if ! command -v caddy >/dev/null 2>&1; then
        curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
        curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | tee /etc/apt/sources.list.d/caddy-stable.list
        apt-get update
        apt-get install -y caddy
    fi
    sed "s|{{DOMAIN}}|$DOMAIN|g" "$APP_DIR/deploy/Caddyfile" > /etc/caddy/Caddyfile
    systemctl restart caddy
fi

echo "==> Запускаю rag-agent"
systemctl restart rag-agent.service
sleep 2
systemctl --no-pager status rag-agent.service || true

cat <<EOF

============================================================
Готово.

Проверьте /etc/systemd/system/rag-agent.service и $APP_DIR/.env.
Не забудьте сменить AUTH_PASSWORD и подставить OPENROUTER_API_KEY.

Логи:           journalctl -u rag-agent -f
Файл логов:     $DATA_DIR/logs/rag.log
Перезапуск:     systemctl restart rag-agent
Локально:       curl -u admin:PASSWORD http://127.0.0.1:8000/api/health
============================================================
EOF
