# RAG Agent

Корпоративный RAG-помощник по документам с регистрацией пользователей и
админ-панелью.

Админ загружает документы (PDF, DOCX, XLSX, PPTX), пользователи регистрируются
и после одобрения админа могут общаться с агентом, который отвечает по
содержимому строго с цитатами на источник (имя файла + страница / лист / слайд).

Стек:

- **FastAPI + Uvicorn** — HTTP API + статический фронтенд
- **PyMuPDF / mammoth / openpyxl / python-pptx** — парсинг
- **Tesseract OCR (rus+eng)** — fallback для сканов PDF
- **Yandex AI Studio embeddings** (`text-search-doc` / `text-search-query`,
  асимметричные) — по умолчанию. Локальный BGE-M3 — опциональный fallback.
- **FAISS IndexFlatIP** — векторный поиск (всё в памяти, ~5 ms)
- **DeepSeek через OpenRouter** — LLM
- **LangGraph `create_react_agent`** — агент с тулом `search_documents`,
  чтобы делать 1–2 поиска за вопрос вместо одного жирного контекста
- **SQLite** — метаданные, история чатов, пользователи, сессии, audit-log
- **Argon2id + cookie sessions** — регистрация, login, рабочая админ-панель
- **HTML/JS** — фронт без сборки (Inter + marked.js + DOMPurify)

Полностью один процесс на одном сервере. Никаких внешних векторных БД,
очередей и микросервисов.

## Аутентификация и роли

- **Регистрация открыта** (можно отключить через `ALLOW_PUBLIC_REGISTRATION=false`).
- Новый пользователь после регистрации получает статус **pending** и не может
  пользоваться чатом, пока админ не одобрит его в админ-панели.
- **Роли**: `admin` (всё, включая загрузку/удаление документов и управление
  пользователями) и `user` (только чат и собственная история).
- **Сессии** в httpOnly cookie с серверным хранением — можно отозвать любую
  через UI или БД.
- **Bootstrap первого админа** через `ADMIN_BOOTSTRAP_EMAIL`/`ADMIN_BOOTSTRAP_PASSWORD`
  при первом старте сервиса. После — смените пароль через UI и удалите эти
  переменные из .env.
- **Защита от взлома**: Argon2id хеши, rate limit на login/register, временный
  лок аккаунта после 5 неудач, audit-log всех значимых событий
  (login_success/fail, approve/reject, role_change, lockout, password_change,
  delete_user).
- **CSRF**: SameSite=Lax cookie + кастомный заголовок `X-Requested-With: fetch`
  на всех state-changing запросах.
- **Security headers**: HSTS, CSP (`default-src 'self'` с whitelist для
  jsdelivr и Google Fonts), X-Frame-Options DENY, Referrer-Policy
  strict-origin-when-cross-origin, Permissions-Policy.

Страницы фронта:
- `/login` — вход
- `/register` — регистрация (если включена)
- `/pending` — экран ожидания одобрения, авто-проверка раз в 15 секунд
- `/` — основное приложение чата
- `/admin` — админ-панель (только для роли admin)

API auth/admin:
- `POST /api/auth/register` — регистрация
- `POST /api/auth/login` — login
- `POST /api/auth/logout` — logout (требует X-Requested-With)
- `GET /api/auth/me` — кто я
- `POST /api/auth/change-password` — смена своего пароля
- `GET /api/admin/users` — список пользователей
- `POST /api/admin/users/{id}/approve` — одобрить, выдать роль
- `POST /api/admin/users/{id}/deactivate` — деактивировать
- `POST /api/admin/users/{id}/role` — сменить роль
- `POST /api/admin/users/{id}/unlock` — снять временный лок
- `POST /api/admin/users/{id}/reset-password` — задать новый пароль
- `DELETE /api/admin/users/{id}` — удалить
- `GET /api/admin/audit` — журнал событий

## Сколько ресурсов нужно

| Конфиг сервера | EMBEDDING_PROVIDER | Подходит? |
|---|---|---|
| **1 GB RAM, 1 vCPU, 25 GB SSD** | yandex (default) | ✅ Достаточно для 1-2 пользователей. |
| **2 GB RAM, 1 vCPU, 40 GB SSD** | yandex | ✅ С запасом. Рекомендуется. |
| **4 GB RAM, 2 vCPU, 40 GB NVMe** | bge (e5-small) | ✅ Локальный fallback без интернета. |
| **8 GB RAM, 2-4 vCPU, 40+ GB SSD** | bge (BGE-M3) | ✅ Лучшее локальное качество. |
| **2 GB RAM, BGE-M3** | bge | ❌ OOM при OCR/индексации. |

С Yandex эмбеддингами уходит ~2 GB RAM (модель в памяти не висит) и
~3 GB диска (нет torch). Рекомендуется тариф **Hetzner CX22 €3.49/мес**
(2 vCPU, 4 GB) или **DigitalOcean s-1vcpu-2gb** ($14/мес).

Цена Yandex embeddings ≈ копейки в месяц на 1-2 пользователей: ~300 тыс. токенов
один раз на индексацию 30-40 документов и ~70 тыс. токенов в месяц на запросы.

Если Yandex не подходит (нет интернета, GDPR, желание полной автономии) —
переключитесь на локальный BGE-M3:

```bash
# 1. Доустановить тяжёлые зависимости
pip install -r requirements-bge-fallback.txt

# 2. В .env: EMBEDDING_PROVIDER=bge

# 3. Переиндексировать корпус
python -m scripts.reindex --yes
```

## Быстрый старт на сервере (Ubuntu 24.04 LTS)

Минимально рекомендуемый Droplet: **DigitalOcean s-4vcpu-8gb** (8 GB RAM,
4 vCPU, 160 GB SSD), Frankfurt/Amsterdam. Альтернатива дешевле — **Hetzner CX32**.

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
# Минимум:
#   OPENROUTER_API_KEY=sk-or-v1-...
#   ADMIN_BOOTSTRAP_EMAIL=you@example.com
#   ADMIN_BOOTSTRAP_PASSWORD=сильный_пароль_минимум_10_символов
# (после первого входа поменяйте пароль через UI и обнулите эти поля)

# 5. Перезапустите
sudo systemctl restart rag-agent

# 6. Проверьте
curl http://127.0.0.1:8000/api/health
```

Зайдите в браузере на `https://rag.example.com`, войдите под bootstrap-админом,
загрузите документы через правую панель и пригласите коллег регистрироваться по
адресу `https://rag.example.com/register`. После регистрации одобрьте их в
`/admin`.

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
# впишите OPENROUTER_API_KEY, ADMIN_BOOTSTRAP_EMAIL, ADMIN_BOOTSTRAP_PASSWORD
# Для локального http:// поставьте SESSION_COOKIE_SECURE=false

uvicorn main:app --reload
# открыть http://127.0.0.1:8000 — будет редирект на /login
```

## API

Все эндпоинты (кроме `/api/auth/login`, `/register`, `/api/health`) требуют
валидной cookie-сессии. Все state-changing запросы (POST/DELETE) требуют
заголовок `X-Requested-With: fetch` (CSRF).

Документы — только для роли `admin` (загрузка/удаление). Чтение — все
авторизованные.

| Метод | Путь | Роль | Назначение |
|---|---|---|---|
| POST | `/api/auth/register` | — | Регистрация нового пользователя (pending) |
| POST | `/api/auth/login` | — | Login, ставит httpOnly-cookie |
| POST | `/api/auth/logout` | user | Завершить сессию |
| GET | `/api/auth/me` | user | Текущий пользователь |
| POST | `/api/auth/change-password` | user | Сменить свой пароль |
| GET | `/api/admin/users` | admin | Список пользователей |
| POST | `/api/admin/users/{id}/approve` | admin | Одобрить + выдать роль |
| POST | `/api/admin/users/{id}/deactivate` | admin | Деактивировать |
| POST | `/api/admin/users/{id}/role` | admin | Сменить роль |
| POST | `/api/admin/users/{id}/unlock` | admin | Снять временный лок |
| POST | `/api/admin/users/{id}/reset-password` | admin | Принудительный сброс пароля |
| DELETE | `/api/admin/users/{id}` | admin | Удалить пользователя и его сессии |
| GET | `/api/admin/audit` | admin | Журнал событий |
| POST | `/api/documents` | admin | Загрузка файлов |
| GET | `/api/documents` | user | Список документов |
| DELETE | `/api/documents/{id}` | admin | Удалить документ |
| POST | `/api/chat` | user | Сообщение в чат |
| GET | `/api/conversations` | user | Свои диалоги (admin — все) |
| POST | `/api/conversations` | user | Создать диалог |
| DELETE | `/api/conversations/{id}` | user | Удалить свой диалог |
| GET | `/api/health` | — | Health check |

### Пример (curl с сессией)

```bash
# Login и сохраняем cookies
curl -c jar.txt -X POST -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"СильныйПароль42"}' \
  http://127.0.0.1:8000/api/auth/login

# Загрузка документа (требует cookies + CSRF header)
curl -b jar.txt -H "X-Requested-With: fetch" \
  -F "files=@manual.pdf" \
  http://127.0.0.1:8000/api/documents

# Чат
curl -b jar.txt -H "X-Requested-With: fetch" -H "Content-Type: application/json" \
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
  поиск, если первая выдача не отвечает на вопрос. Контекст не пухнет: тул
  каждый раз возвращает только top-K свежих чанков, не накапливает.
- **Качество retrieval встроено внутрь tool, а не возложено на LLM.** Один
  вызов `search_documents` под капотом включает (всё опционально, см. `.env`):
  - **Multi-query** — LLM генерирует 2 альтернативные формулировки, выдачи
    объединяются через RRF;
  - **HyDE** — для определительных запросов («что такое X») LLM генерирует
    гипотетический фрагмент-ответ, его эмбеддинг идёт в поиск вместе с запросом;
  - **Hybrid dense + BM25** — RRF поверх FAISS и BM25 (BM25 всегда выручает
    запросы по датам, кодам, именам);
  - **Entity routing** — запросы из преимущественно дат/артикулов/кодов
    автоматически идут BM25-only, без шумного семантического поиска;
  - **Reformulate-on-low-score** — если средний score топа ниже порога, LLM
    переформулирует с подсказкой из найденных терминов, и поиск повторяется;
  - **Reranker** — top-N кандидатов реранкятся через основную LLM (DeepSeek)
    либо через локальный cross-encoder (BGE reranker, опц.);
  - **Adaptive top-K** — для перечислений / сравнений возвращается больше
    фрагментов, для длинных конкретных — меньше.
- **Адаптивный чанкинг.** Размер чанков подбирается под тип документа
  (xlsx/csv → 1200, pptx → 1500, длинные PDF → 3000). Опционально доступен
  семантический сплит: между параграфами считается cosine эмбеддингов и
  граница ставится только там, где смысл реально меняется.
- **Smart table extraction для PDF.** Таблицы извлекаются через `pdfplumber`
  как отдельные сегменты в Markdown — строка не размывается между чанками.
- **Пост-сверка ответа.** После генерации мы делаем дешёвый LLM-вызов и
  сверяем утверждения с найденными фрагментами. Если что-то не подтверждено,
  к ответу добавляется явное предупреждение со списком неподтверждённых
  утверждений (никакой автоматической retry — слишком дорого, а явное
  предупреждение даёт пользователю шанс перепроверить).
- **Token-эффективность.** История режется до 8 сообщений, inline-цитаты
  вычищаются из истории перед отправкой в LLM (иначе модель «рециркулирует»
  свои же выдуманные ссылки). Полный текст чанка не уезжает в LLM повторно.
- **FAISS Flat, а не HNSW.** На 30–40 документов — максимум ~5000 векторов.
  Точный поиск занимает миллисекунды, ничего сложнее не нужно.
- **Один воркер uvicorn.** Модель эмбеддингов и FAISS-индекс не имеет смысла
  дублировать. CPU-bound операции выполняются в `run_in_threadpool`.

### Стоимость доп. слоёв

Все улучшения выше «платятся» дополнительными LLM-вызовами:

| Слой | Стоимость на один вопрос | По умолчанию |
|---|---|---|
| Multi-query | +1 LLM вызов на каждый search | ON |
| HyDE | +1 LLM + 1 embedding на definition-запрос | ON |
| Reformulate-on-low | +1 LLM (только при низком score) | ON |
| Reranker (provider=llm) | +1 LLM на каждый search | ON |
| Reranker (provider=ce) | 0 (но +0.5 GB RAM на cross-encoder) | OFF |
| Answer verification | +1 LLM на ответ | ON |

На один вопрос пользователя с одним search'ом это в среднем 3–5 LLM-вызовов
вместо 1 «классических». Для DeepSeek через OpenRouter это $0.005–0.02 за
вопрос. Если нужно дешевле — выключите часть слоёв через `.env` (поиск
останется работоспособным с любым подмножеством).

## Структура проекта

```
.
├── main.py                  # FastAPI app, lifespan, security headers, статика
├── config.py                # pydantic-settings из .env
├── requirements.txt
├── .env.example
├── auth/
│   ├── passwords.py         # Argon2id хеширование, валидация силы
│   ├── sessions.py          # cookie httpOnly, sha256(token) в БД
│   ├── dependencies.py      # require_user / require_admin / csrf_check
│   └── router.py            # /api/auth/* + rate-limit (slowapi)
├── api/
│   ├── documents.py         # upload (admin) / list / delete (admin) + фон. обработка
│   ├── chat.py              # POST /api/chat (изоляция диалогов по user_id)
│   ├── conversations.py
│   └── admin.py             # /api/admin/users/* + /api/admin/audit
├── parsers/                 # один модуль на формат
│   ├── pdf_parser.py        # PyMuPDF + OCR fallback + pdfplumber для таблиц
│   ├── docx_parser.py       # mammoth → markdown
│   ├── xlsx_parser.py       # маленькие листы целиком, большие — построчно
│   ├── pptx_parser.py
│   └── router.py            # python-magic + расширение → нужный парсер
├── chunking/chunker.py      # adaptive size + structural / semantic split
├── embeddings/bge_m3.py     # singleton, FlagEmbedding или sentence-transformers
├── search/
│   ├── faiss_index.py       # FAISS + hybrid SearchService (dense+BM25+RRF)
│   ├── entity_detector.py   # daters/codes/intents → routing & adaptive top-K
│   ├── query_expansion.py   # multi-query + HyDE + reformulate
│   └── reranker.py          # LLM / cross-encoder реранкер top-N → top-K
├── llm/
│   ├── prompts.py
│   ├── agent.py             # ReAct-агент с tool search_documents
│   └── verifier.py          # пост-сверка утверждений ответа с чанками
├── storage/database.py      # SQLite: documents, chunks, conversations, messages,
│                            # users, sessions, auth_audit
├── static/                  # фронт без сборки
│   ├── index.html, app.js, style.css   # главное приложение
│   ├── login.html, register.html, pending.html, auth.js, auth.css
│   └── admin.html, admin.js
└── deploy/
    ├── install.sh
    ├── rag-agent.service
    ├── Caddyfile
    └── backup.sh
```

## Смена эмбеддинг-провайдера

Эмбеддер выбирается через `EMBEDDING_PROVIDER` в `.env`:

| Провайдер | Размерность | Контекст | RAM | Стоимость |
|---|---|---|---|---|
| `yandex` (default) | 256 | ~2048 токенов | ~50 MB | копейки/мес |
| `bge` (BGE-M3) | 1024 | 8192 токенов | ~2 GB | 0 |
| `bge` (e5-small) | 384 | 512 токенов | ~500 MB | 0 |

**Размерности несовместимы.** Если меняете провайдер на работающем сервисе:

```bash
cd /opt/rag_agent_v1
# 1. Поправить EMBEDDING_PROVIDER в .env
sudo -u rag nano .env
# 2. Если переходите на bge — доустановить:
sudo -u rag .venv/bin/pip install -r requirements-bge-fallback.txt
# 3. Перепрогнать существующие чанки через новую модель
sudo -u rag .venv/bin/python -m scripts.reindex --yes
# 4. Перезапустить
sudo systemctl restart rag-agent
```

Скрипт `scripts/reindex.py` берёт текст всех чанков из SQLite, прогоняет через
текущий провайдер и пересобирает FAISS-индекс. Файлы документов в
`data/uploads/` не трогаются.

**Асимметричные модели Yandex.** В отличие от BGE-M3 (одна модель и для doc,
и для query), Yandex использует две разные модели — `text-search-doc` для
индексации и `text-search-query` для поиска. Это не ошибка, а специально
обученное под retrieval решение, обычно дающее качество выше симметричных
моделей. Менять местами в `.env` нельзя — поиск просядет.

## Лимиты и эксплуатация

- Размер файла: `MAX_FILE_SIZE_MB=50`
- Документов суммарно: `MAX_DOCUMENTS=100`
- История в LLM: последние `MAX_HISTORY_MESSAGES=8` сообщений
- Сессия: `SESSION_LIFETIME_DAYS=7`, sliding (обновляется при активности)
- Brute force: `LOGIN_MAX_FAILED_ATTEMPTS=5` → лок на `LOGIN_LOCKOUT_MINUTES=15`
- Rate limit: `RATE_LIMIT_LOGIN_PER_MINUTE=10` / `RATE_LIMIT_REGISTER_PER_HOUR=5`
- Логи: `/data/logs/rag.log` с ротацией 10 MB × 5 файлов + journald
- Бэкапы: `deploy/backup.sh` в `/etc/cron.daily/`

## Безопасность — что сделано

| Угроза | Защита |
|---|---|
| Brute force паролей | Argon2id (~50ms на хеш) + rate limit + лок аккаунта на 15 мин после 5 неудач |
| Кража БД с паролями | Argon2id с уникальной солью (формат `$argon2id$...`) |
| Кража cookie | httpOnly + Secure + SameSite=Lax. В БД хранится только sha256(token), а не сам токен |
| Session fixation | Новый токен генерится на каждый login |
| CSRF | SameSite=Lax + кастомный header `X-Requested-With: fetch` на POST/PUT/DELETE |
| XSS в ответе LLM | DOMPurify санитизирует markdown-вывод; строгий CSP |
| XSS в имени файла / диалога | `textContent` вместо `innerHTML` для пользовательского ввода |
| Path traversal в загрузке | Sanitization имени файла + изоляция в `/data/uploads/{doc_id}/` |
| SQL injection | Только параметризованные запросы |
| Email enumeration | Generic-ошибка при дубликате на регистрации, dummy-verify при отсутствии юзера |
| Timing attack на login | Прогон через verify даже если юзера нет |
| Privilege escalation | Server-side role check на каждом admin-эндпоинте |
| Удаление последнего админа | Заблокировано в API |
| Clickjacking | `X-Frame-Options: DENY` + `frame-ancestors 'none'` |
| MIME sniffing | `X-Content-Type-Options: nosniff` |
| HTTPS downgrade | HSTS (max-age=1 год) когда `SESSION_COOKIE_SECURE=true` |
| Утечка прав через API | Документы — только admin upload/delete; чтение для всех; чаты per-user |
| Audit log | login/logout/approve/reject/role_change/lockout/password_change в `auth_audit` |

## Что НЕ сделано (намеренно)

- **Email-верификация** — требует SMTP. Mitigated: rate-limit на регистрацию + ручной approval.
- **2FA** — для 1-2 пользователей оверкилл; админ может включить в next-итерации (TOTP).
- **Password reset через email** — сейчас reset делает админ через UI.
- **OAuth/SSO** — out of scope.
- **Per-user изоляция документов** — by design: документы корпоративные, общие.
  Только админ загружает; users только спрашивают и видят свои чаты.
- Загрузки изолированы в `/data/uploads/{document_id}/`. Имя файла берётся как
  есть, путь не используется как параметр в shell-командах.

## Лицензия

MIT
