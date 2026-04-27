#!/usr/bin/env bash
# Простой бэкап /data в tar.gz. Запускайте из cron раз в сутки.
# Например, в /etc/cron.daily/rag-backup:
#   #!/bin/sh
#   /opt/rag_agent_v1/deploy/backup.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/rag-agent}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"

mkdir -p "$BACKUP_DIR"
TS=$(date +%Y%m%d-%H%M%S)
TARGET="$BACKUP_DIR/rag-data-$TS.tar.gz"

tar -czf "$TARGET" -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"
echo "Backup: $TARGET ($(du -h "$TARGET" | cut -f1))"

# Чистим старые
find "$BACKUP_DIR" -name "rag-data-*.tar.gz" -mtime +"$RETENTION_DAYS" -delete
