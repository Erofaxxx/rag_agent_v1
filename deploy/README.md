# Deploy

Скрипты для развёртывания на одном Ubuntu 24.04 LTS Droplet.

## Файлы

- `install.sh` — основной установщик (apt + venv + systemd + Caddy).
- `rag-agent.service` — systemd-юнит для приложения.
- `Caddyfile` — реверс-прокси с автоматическим HTTPS.
- `backup.sh` — ежедневный tar.gz `/data` в `/var/backups/rag-agent`.

## Шаги

1. Создать DigitalOcean Droplet `s-4vcpu-8gb`, Ubuntu 24.04, регион Frankfurt
   или Amsterdam. Пробросить домен `rag.example.com → IP_droplet` (A-запись).
2. Зайти по SSH под root.
3. Склонировать репозиторий в `/opt/rag_agent_v1`.
4. Запустить `sudo DOMAIN=rag.example.com bash deploy/install.sh`.
5. Отредактировать `/opt/rag_agent_v1/.env` — поставить `OPENROUTER_API_KEY`
   и `AUTH_PASSWORD`.
6. `sudo systemctl restart rag-agent`.

## Бэкапы

```bash
sudo cp /opt/rag_agent_v1/deploy/backup.sh /etc/cron.daily/rag-backup
sudo chmod +x /etc/cron.daily/rag-backup
```

Опционально — отправлять архивы в DO Spaces / S3:

```bash
# в /etc/cron.daily/rag-backup, после backup.sh:
LATEST=$(ls -t /var/backups/rag-agent/*.tar.gz | head -1)
s3cmd put "$LATEST" s3://your-bucket/rag-backups/
```

## Проверка после установки

```bash
systemctl status rag-agent
journalctl -u rag-agent -f

# health
curl -u admin:PASS http://127.0.0.1:8000/api/health
```

## Обновление

```bash
cd /opt/rag_agent_v1
sudo -u rag git pull
sudo -u rag .venv/bin/pip install -r requirements.txt
sudo systemctl restart rag-agent
```

## Откат

`/data` сохраняется бэкапами. Чтобы откатить код:

```bash
cd /opt/rag_agent_v1
sudo -u rag git checkout <commit_sha>
sudo systemctl restart rag-agent
```
