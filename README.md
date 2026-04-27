# RAG Agent

Корпоративный RAG-помощник по документам.

Один пользователь загружает 30–40 документов (PDF, DOCX, XLSX, PPTX) и
переписывается с агентом, который отвечает по содержимому строго с цитатами
на источник (имя файла + страница / лист / слайд).

Стек:

- **FastAPI + Uvicorn** — HTTP API + статический фронтенд
- **PyMuPDF / mammoth / openpyxl / python-pptx** — парсинг
- **Tesseract OCR (rus+eng)** — fallback для сканов PDF
- **BGE-M3** (`BAAI/bge-m3`) — эмбеддинги в FP16 на CPU
- **FAISS IndexFlatIP** — векторный поиск (всё в памяти, ~5 ms)
- **DeepSeek через OpenRouter** — LLM
- **LangGraph `create_react_agent`** — агент с тулом `search_documents`,
  чтобы делать 1–2 поиска за вопрос вместо одного жирного контекста
- **SQLite** — метаданные, история чатов
- **HTML/JS** — минимальный фронтенд с drag-and-drop папок и чатом

Полностью один процесс на одном сервере. Никаких внешних векторных БД,
очередей и микросервисов.

## Быстрый старт на сервере (Ubuntu 24.04 LTS)

Минимально рекомендуемый Droplet: **DigitalOcean s-4vcpu-8gb** (8 GB RAM,
4 vCPU, 160 GB SSD), Frankfurt/Amsterdam.

```bash
# 1. Клонируйте репозиторий
sudo mkdir -p /opt
sudo git clone https://github.com/Erofaxxx/rag_agent_v1 /opt/rag_agent_v1

# 2. Создайте пользователя rag и каталог /data, отдайте ему права
sudo useradd -m -s /bin/bash rag
sudo mkdir -p /data && sudo chown rag:rag /data /opt/rag_agent_v1

# 3. Запустите установщик
cd /opt/rag_agent_v1
sudo DOMAIN=rag.example.com bash deploy/install.sh
# DOMAIN можно не задавать — тогда Caddy не настроится, и сервис будет
# слушать только на 127.0.0.1:8000.

# 4. Отредактируйте /opt/rag_agent_v1/.env
sudo -u rag nano /opt/rag_agent_v1/.env
# Минимум: OPENROUTER_API_KEY и AUTH_PASSWORD

# 5. Перезапустите
sudo systemctl restart rag-agent

# 6. Проверьте
curl -u admin:ВАШ_ПАРОЛЬ http://127.0.0.1:8000/api/health
```

Зайдите в браузере на `https://rag.example.com` (или `http://IP:8000`),
залогиньтесь, перетащите папку с документами — и общайтесь.

## Что устанавливает install.sh

- Системные пакеты: `python3.11`, `tesseract-ocr` (`rus+eng`), `poppler-utils`,
  `libreoffice`, `libmagic1`, `caddy` (если задан `DOMAIN`).
- Python venv в `/opt/rag_agent_v1/.venv` с зависимостями из
  [requirements.txt](requirements.txt).
- Скачивает модель `BAAI/bge-m3` в кэш `~rag/.cache/huggingface` (~2.3 GB).
  При первом запросе она будет уже на диске — старт сервиса быстрый.
- systemd-юнит `rag-agent.service` (auto-restart, логи в journald + файл).
- Caddyfile с автоматическим HTTPS (Let's Encrypt) при заданном `DOMAIN`.

## Локальная разработка (macOS / Linux)

```bash
git clone https://github.com/Erofaxxx/rag_agent_v1
cd rag_agent_v1
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Системные зависимости (macOS):
brew install tesseract tesseract-lang poppler libmagic libreoffice

cp .env.example .env
# впишите OPENROUTER_API_KEY и AUTH_PASSWORD

uvicorn main:app --reload
# открыть http://127.0.0.1:8000 — браузер спросит admin / ваш пароль
```

## API

Все эндпоинты под HTTP Basic Auth (`AUTH_USERNAME` / `AUTH_PASSWORD` из `.env`).

| Метод   | Путь                                | Назначение                          |
|---------|-------------------------------------|-------------------------------------|
| POST    | `/api/documents`                    | Загрузка одного или нескольких файлов |
| GET     | `/api/documents`                    | Список документов со статусами      |
| GET     | `/api/documents/{id}/status`        | Статус обработки (для polling)      |
| DELETE  | `/api/documents/{id}`               | Удаление документа и его чанков     |
| POST    | `/api/chat`                         | Сообщение в чат, ответ с цитатами   |
| GET     | `/api/conversations`                | Список диалогов                     |
| POST    | `/api/conversations`                | Создать новый диалог                |
| GET     | `/api/conversations/{id}`           | История одного диалога              |
| DELETE  | `/api/conversations/{id}`           | Удалить диалог                      |
| GET     | `/api/health`                       | Health check                        |

### Пример

```bash
# Загрузка
curl -u admin:PASS -F "files=@manual.pdf" -F "files=@report.xlsx" \
  http://127.0.0.1:8000/api/documents

# Чат
curl -u admin:PASS -H "Content-Type: application/json" \
  -d '{"message": "Что сказано про сроки в договоре?"}' \
  http://127.0.0.1:8000/api/chat
```

Ответ `/api/chat`:

```json
{
  "conversation_id": 1,
  "message_id": 2,
  "answer": "Срок действия договора — 12 месяцев [contract.pdf, стр. 3]...",
  "cited_chunks": [
    {
      "chunk_id": 42, "document_id": 1,
      "filename": "contract.pdf", "page_number": 3,
      "score": 0.81, "snippet": "..."
    }
  ]
}
```

## Архитектура и почему так

```
┌──────────┐   upload   ┌──────────┐
│ Browser  │──────────▶│ FastAPI  │── BackgroundTask ──▶ parse → chunk → embed → FAISS + SQLite
│ (HTML/JS)│            │   API    │
│          │   chat     │          │── ReAct Agent ────▶ search_documents tool ──▶ FAISS → SQLite chunks
└──────────┘◀──────────└──────────┘                              │
                                                                  ▼
                                                          OpenRouter / DeepSeek
```

- **Агент, а не плоский RAG.** Используется `langgraph.prebuilt.create_react_agent`
  с одним инструментом `search_documents`. Агент сам решает, делать ли ещё один
  поиск с переформулированным запросом, если первая выдача не отвечает на вопрос.
  При этом контекст не пухнет: тул каждый раз возвращает только top-7 свежих
  чанков, а не накапливает.
- **Token-эффективность.** История диалога режется до 8 сообщений, в каждом
  ответе — список процитированных чанков с короткими сниппетами. Полный текст
  чанка не уезжает в LLM повторно — он лежит в SQLite.
- **FAISS Flat, а не HNSW.** На 30–40 документов — максимум ~5000 векторов.
  Точный поиск занимает миллисекунды, ничего сложнее не нужно.
- **Один воркер uvicorn.** Модель эмбеддингов (~2 GB RAM) и FAISS-индекс не
  имеет смысла дублировать; пользователь один. CPU-bound операции (эмбеддинг)
  выполняются в `run_in_threadpool`, чтобы не блокировать event loop.

## Структура проекта

```
.
├── main.py                  # FastAPI app, lifespan, статика
├── config.py                # pydantic-settings из .env
├── requirements.txt
├── .env.example
├── api/                     # эндпоинты
│   ├── auth.py              # HTTP Basic
│   ├── documents.py         # upload / list / delete + фоновая обработка
│   ├── chat.py              # POST /api/chat
│   └── conversations.py
├── parsers/                 # один модуль на формат
│   ├── pdf_parser.py        # PyMuPDF + OCR fallback
│   ├── docx_parser.py       # mammoth → markdown
│   ├── xlsx_parser.py       # маленькие листы целиком, большие — построчно
│   ├── pptx_parser.py
│   └── router.py            # python-magic + расширение → нужный парсер
├── chunking/chunker.py      # рекурсивный сплиттер по сепараторам
├── embeddings/bge_m3.py     # singleton, FlagEmbedding или sentence-transformers
├── search/faiss_index.py    # FAISS + SearchService с опциональным BM25
├── llm/
│   ├── prompts.py
│   └── agent.py             # ReAct-агент с tool search_documents
├── storage/database.py      # SQLite (WAL, foreign keys)
├── static/                  # минимальный фронт (drag-n-drop + чат)
└── deploy/
    ├── install.sh
    ├── rag-agent.service
    ├── Caddyfile
    └── backup.sh
```

## Лимиты и эксплуатация

- Размер файла: `MAX_FILE_SIZE_MB=50`
- Документов суммарно: `MAX_DOCUMENTS=100`
- История в LLM: последние `MAX_HISTORY_MESSAGES=8` сообщений
- Логи: `/data/logs/rag.log` с ротацией 10 MB × 5 файлов + journald
- Бэкапы: `deploy/backup.sh` в `/etc/cron.daily/`

## Что НЕ сделано (намеренно)

См. [ТЗ §14](#) — никакого Qdrant, реранкера, Docker, Celery, мультиюзерности.
LLM-стриминг можно добавить второй итерацией: сейчас ответ возвращается одним
JSON.

## Безопасность

- HTTP Basic — по сути один пароль. Закрыто HTTPS через Caddy. Достаточно для
  одного пользователя; не годится для публичной системы.
- API-ключ OpenRouter — только в `.env`, **не коммитьте**.
- Загрузки изолированы в `/data/uploads/{document_id}/`. Имя файла берётся как
  есть, путь не используется как параметр в shell-командах.

## Лицензия

MIT
