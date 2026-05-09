from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    OPENROUTER_API_KEY: str
    LLM_MODEL: str = "deepseek/deepseek-chat"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 8000
    OPENROUTER_HTTP_REFERER: str = "http://localhost"
    OPENROUTER_X_TITLE: str = "RAG Agent"

    # Auth / sessions
    SESSION_COOKIE_NAME: str = "rag_session"
    SESSION_LIFETIME_DAYS: int = 7
    SESSION_COOKIE_SECURE: bool = True  # в dev (http://localhost) ставьте False
    SESSION_COOKIE_SAMESITE: str = "lax"

    # Bootstrap первого админа при первом запуске. Если в БД ещё нет ни одного
    # пользователя, создаётся админ с этими credentials. Дальше — менять пароль
    # через UI и больше не использовать ADMIN_BOOTSTRAP_PASSWORD.
    ADMIN_BOOTSTRAP_EMAIL: str = ""
    ADMIN_BOOTSTRAP_PASSWORD: str = ""

    # CORS — список доменов через запятую. По умолчанию — same-origin only.
    # Пример: CORS_ORIGINS=https://rag.example.com
    CORS_ORIGINS: str = ""

    # Регистрация
    ALLOW_PUBLIC_REGISTRATION: bool = True
    PASSWORD_MIN_LENGTH: int = 10

    # Защита от перебора
    LOGIN_MAX_FAILED_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15
    RATE_LIMIT_LOGIN_PER_MINUTE: int = 10
    RATE_LIMIT_REGISTER_PER_HOUR: int = 5
    # Лимит на /api/chat — каждый авторизованный запрос идёт в LLM (платно).
    RATE_LIMIT_CHAT_PER_MINUTE: int = 30
    # Лимит на загрузку документов — каждая загрузка спавнит парсер и эмбеддинг
    # пайплайн, который активно жрёт CPU/RAM/квоту эмбеддингов.
    RATE_LIMIT_UPLOAD_PER_MINUTE: int = 20
    # Доверять X-Forwarded-For / X-Real-IP. По умолчанию OFF: атакующий иначе
    # подделывает заголовок и обходит per-IP rate limiter и audit-log с разными
    # «адресами». Включать ТОЛЬКО когда сервис стоит за trusted reverse-proxy
    # (Caddy/nginx), который перезаписывает заголовок на реальный client IP.
    TRUST_PROXY_HEADERS: bool = False

    # Paths
    DATA_DIR: str = "./data"

    # Embedding provider: 'yandex' (по умолчанию, через AI Studio) или 'bge'
    # (локальный BGE-M3 / multilingual-e5; требует requirements-bge-fallback.txt).
    EMBEDDING_PROVIDER: str = "yandex"

    # ---- Yandex AI Studio ----
    # Получить API-ключ: console.yandex.cloud → IAM → Сервисные аккаунты →
    # создать аккаунт с ролью ai.languageModels.user → API-ключ
    YANDEX_API_KEY: str = ""
    YANDEX_FOLDER_ID: str = ""
    # Асимметричные модели — НЕ путать местами! При индексации зовётся doc-модель,
    # при поиске — query-модель. Это критично для качества retrieval.
    YANDEX_EMBEDDING_DOC_MODEL: str = "text-search-doc"
    YANDEX_EMBEDDING_QUERY_MODEL: str = "text-search-query"
    # Размерность вектора. По умолчанию у Yandex — 256. Можно понизить (64-128)
    # для экономии места и ускорения, но качество падает. 0 = использовать default.
    YANDEX_EMBEDDING_DIMENSIONS: int = 0
    # Защита от throttle. Yandex не публикует точный RPM, поэтому делаем
    # консервативный лимит на стороне клиента. По дефолту лимит ~10 RPS на
    # папку, для большинства инсталляций 600 RPM (=10 RPS) безопасно.
    YANDEX_EMBEDDING_RPM: int = 600
    # Параллелизм при индексации документа. SDK не поддерживает batch-эндпоинт
    # (один text на запрос), но запросы можно слать параллельно — RPS-бюджет
    # тратит slot-based throttle ниже. CONCURRENCY=8 при RPM=600 (≈10 RPS)
    # даёт 3-5× ускорение индексации без риска получить 429.
    YANDEX_EMBEDDING_CONCURRENCY: int = 8
    # Yandex embedding context — 2048 токенов (≈ 6000-8000 русских символов).
    # Чанк длиннее — будет обрезан сервером, потеряем контекст.
    YANDEX_EMBEDDING_MAX_CHARS: int = 6000

    # ---- BGE-M3 / sentence-transformers fallback ----
    # Используются ТОЛЬКО когда EMBEDDING_PROVIDER=bge.
    # Для серверов с 4 GB RAM подойдёт `intfloat/multilingual-e5-small` (~500 MB):
    #   EMBEDDING_MODEL=intfloat/multilingual-e5-small
    #   EMBEDDING_QUERY_PREFIX=query:
    #   EMBEDDING_PASSAGE_PREFIX=passage:
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_USE_FP16: bool = True
    EMBEDDING_QUERY_PREFIX: str = ""
    EMBEDDING_PASSAGE_PREFIX: str = ""
    EMBEDDING_MAX_LENGTH: int = 8192

    # Chunking (в символах; русский текст ~3-4 символа на токен)
    CHUNK_SIZE: int = 2400
    CHUNK_OVERLAP: int = 400
    # Адаптивный размер чанков под тип документа: для xlsx/csv/pptx нужны
    # короткие плотные чанки, для длинных PDF выгоднее большие. Если выключить,
    # все документы режутся одним CHUNK_SIZE/CHUNK_OVERLAP.
    CHUNK_ADAPTIVE: bool = True
    # Семантический чанкинг: между параграфами считаем cosine соседних
    # эмбеддингов и не сливаем те, между которыми смысл резко меняется.
    # Дороже по эмбеддингам (≈ +1 запрос на параграф), но снимает разрывы
    # «посреди объяснения». По умолчанию выключено — включай, если индексация
    # один раз на корпус, а качество критично.
    CHUNK_SEMANTIC: bool = False
    # Порог cosine между соседними параграфами: ниже — граница чанка.
    CHUNK_SEMANTIC_THRESHOLD: float = 0.55

    # ---- Search ----
    SEARCH_TOP_K: int = 7
    # Гибридный поиск (dense + BM25, объединение через RRF). По умолчанию
    # включён: BM25 сильно вытаскивает запросы по точным терминам, датам
    # и кодам, которые в эмбеддингах размываются.
    SEARCH_USE_BM25: bool = True
    # Адаптивный k: для коротких/перечислительных запросов берём больше
    # фрагментов, для длинных конкретных — меньше.
    SEARCH_ADAPTIVE_K: bool = True
    # Реранкер поверх top-N кандидатов из FAISS/BM25. Драматически чистит
    # топ от мусора. Провайдеры:
    #  off  — не делать
    #  llm  — реранк через основную LLM (DeepSeek), zero install, +1 запрос
    #  ce   — локальный cross-encoder (нужен sentence-transformers + torch)
    RERANKER_PROVIDER: str = "llm"
    # Сколько кандидатов из retrieval'а пойдёт в реранкер.
    RERANKER_CANDIDATES: int = 20
    # Cross-encoder модель (если RERANKER_PROVIDER=ce). bge-reranker-v2-m3 —
    # мультиязычный, ~570 MB, хорошо работает на русском и английском.
    RERANKER_CE_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # Multi-query: LLM генерирует доп. формулировки и мы объединяем выдачи
    # через RRF. Сильно помогает на коротких/неоднозначных запросах.
    SEARCH_MULTI_QUERY: bool = True
    SEARCH_MULTI_QUERY_COUNT: int = 3  # включая оригинал
    # HyDE: для «определительных» запросов (что такое X, определение Y) LLM
    # генерирует короткий гипотетический ответ, и мы ищем по ЕГО эмбеддингу
    # вместо/в добавок к запросу.
    SEARCH_HYDE: bool = True
    # Если средний score топа ниже порога — внутри tool делаем 1 повторный
    # поиск с переформулированным запросом. Порог в шкале inner-product
    # на L2-нормализованных векторах (≈ cosine).
    SEARCH_LOW_SCORE_THRESHOLD: float = 0.45
    SEARCH_REFORMULATE_ON_LOW: bool = True
    # Запросы, состоящие в основном из дат/кодов/артикулов/имён собственных,
    # выгоднее искать BM25-only — embedding на таких токенах стабильно врёт.
    SEARCH_ENTITY_BM25_FALLBACK: bool = True

    # ---- Answer quality ----
    # После генерации ответа просим LLM сверить утверждения с найденными
    # чанками. Если есть неподтверждённое — добавляем в ответ короткое
    # предупреждение. Без retry — слишком дорого. +1 LLM вызов на вопрос.
    ANSWER_VERIFICATION: bool = True

    # Лимит инструментальных вызовов агента за один вопрос юзера. Hard cap —
    # после исчерпания tool вернёт «лимит достигнут», и LLM должна отвечать
    # уже без поиска. Защита от runaway-итераций.
    MAX_TOOL_CALLS_PER_QUESTION: int = 5

    # Limits
    MAX_FILE_SIZE_MB: int = 50
    MAX_DOCUMENTS: int = 100
    MAX_HISTORY_MESSAGES: int = 8

    # Хранить оригиналы загруженных файлов после парсинга? По умолчанию — да.
    #
    # Зачем хранить:
    # - Перепарсинг при обновлении парсеров (новый PyMuPDF, фикс OCR-бага и т.п.)
    # - Перечанкинг при смене размера/сепараторов
    # - Скачивание оригинала пользователем
    # - Compliance / аудит
    # - Восстановление при повреждении индекса
    #
    # Зачем выключать (=false):
    # - Очень ограниченный диск (но 100 файлов × 50 MB = max 5 GB на юзера)
    # - Жёсткие требования privacy «не хранить оригиналы»
    KEEP_ORIGINAL_FILES: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # OCR
    OCR_LANGUAGES: str = "rus+eng"
    OCR_MIN_CHARS_PER_PAGE: int = 100

    # ---- Security research: poisoning / backdoor defenses ----
    # Все флаги по умолчанию выключены — поведение агента не меняется,
    # пока кто-то осознанно не включит защиты в .env. Это позволяет
    # сначала прогнать базовый сценарий (агент работает как раньше),
    # потом снять метрики атак, и только затем включать защиты слой за слоем.

    # L0: corpus consistency / near-duplicate document detection at ingest.
    # Generic защита против гибридного backdoor: атакующий клонирует
    # легитимный документ корпуса и вшивает 1-2 раздела с target-фразой.
    # На уровне отдельного chunk такой payload не отличим (тематика та же,
    # лексика нормальная), но на уровне документа видно: 70-95% chunks
    # почти идентичны (cosine ≥ similarity_threshold) другому документу из
    # индекса, плюс есть 1-2 inserted chunks без аналогов. Стоимость:
    # дополнительный FAISS-поиск на каждый chunk нового документа при ingest;
    # без LLM. Действие: 'off' / 'warn' / 'drop' (документ блокируется).
    DEFENSE_L0_CORPUS_CONSISTENCY: str = "off"
    # Cosine, при котором пара chunks считается «почти идентичной». 0.92 — эмпирически
    # хорошо отделяет «копия» от «просто близкая тема» на embeddings-моделях
    # типа BGE-M3 / Yandex text-search. Снижение → больше FP, рост → пропускает
    # «слегка перефразированные» клоны.
    DEFENSE_L0_SIMILARITY_THRESHOLD: float = 0.92
    # Доля chunks нового документа, чьи лучшие соседи лежат в одном и том же
    # существующем документе. 0.7 = «70% чанков нового — почти-копии чанков
    # одного и того же файла, плюс есть inserted разделы».
    DEFENSE_L0_DUPLICATE_RATIO_THRESHOLD: float = 0.7

    # L1: sanitization чанков на ingest. Ищет prompt-injection паттерны,
    # role-switch, фейковые цитаты в тексте, подозрительные unicode-блоки.
    # Действие: 'off' (только лог), 'warn' (помечает risk_score), 'drop'
    # (отбрасывает чанки выше порога — НЕ ставь в prod без аудита).
    DEFENSE_L1_SANITIZE: str = "off"
    DEFENSE_L1_RISK_THRESHOLD: float = 0.6

    # L2: embedding-space anomaly detection. По эмбеддингам чанков считаем
    # центроид документа и расстояние; чанки с z-score выше порога флагуются.
    # Действие аналогично L1.
    DEFENSE_L2_ANOMALY: str = "off"
    DEFENSE_L2_ZSCORE_THRESHOLD: float = 2.5

    # L3: query-ablation детектор trigger-based backdoor'ов (BadNets-style).
    # Generic защита: для каждого «значимого» слова запроса делаем дополнительный
    # retrieval с удалённым словом. Чанк, который выпадает из top-k при удалении
    # одного слова, считается «триггер-активированным» и помечается как
    # подозрительный. В отличие от L1/L4 не зависит от знания шаблонов атаки —
    # ловит любые перефразированные стелс-бэкдоры с триггером в запросе.
    # Стоимость: до DEFENSE_L3_MAX_ABLATIONS дополнительных FAISS-поисков
    # на каждый search_documents вызов; БЕЗ LLM-вызовов.
    # Действие: 'off' / 'warn' (плашка в ответе) / 'drop' (выкидываем chunks).
    DEFENSE_L3_QUERY_ABLATION: str = "off"
    # Trigger threshold: chunk считается подозрительным, если выпал из top-k
    # при доле ≥ threshold ablations. Реальные триггеры обычно состоят из
    # 2-3 редких слов, поэтому удаление 1-2 из них ломает retrieval —
    # 0.5 даёт хороший signal-to-noise для multi-word триггеров.
    DEFENSE_L3_TRIGGER_THRESHOLD: float = 0.5
    # Сколько слов запроса максимум абляровать (топ N по длине). Триггеры
    # обычно содержат редкие/длинные слова, потому 8 хватает на 99% реальных
    # запросов. Каждое ablation = +1 FAISS-вызов, поэтому не задирай.
    DEFENSE_L3_MAX_ABLATIONS: int = 8
    # Минимальная длина core-слова, чтобы считаться кандидатом для ablation.
    # Защита от трат FAISS-вызовов на «и»/«в»/«на».
    DEFENSE_L3_MIN_WORD_LEN: int = 4

    # L4: расширенный верификатор ответа. К текущему verify_answer добавляет
    # проверку injection-паттернов в самих cited chunks — если триггер
    # «прошёл» в LLM, в ответе появится явное предупреждение.
    DEFENSE_L4_STRICT_VERIFIER: bool = False

    # L6: ingest-time LLM-judge на противоречия с существующим корпусом.
    # Закрывает архитектурный пробел: атаки, которые L0/L1/L2 не видят
    # (нет structurного клона, нет regex-сигнатур, мало chunks), но
    # фактически противоречат корпусу — например, документ «новая
    # редакция, отменяющая предыдущую» или одиночный chunk с триггер-
    # утверждением «согласно X лимит снят». Стоимость: +1 LLM-вызов
    # на каждый chunk нового документа, у которого есть «соседи»
    # (cosine ≥ 0.5) в индексе. Действие: 'off' / 'warn' / 'drop'.
    DEFENSE_L6_INGEST_CONTRADICTION: str = "off"
    DEFENSE_L6_SIMILARITY_THRESHOLD: float = 0.5
    DEFENSE_L6_TOP_K_NEIGHBORS: int = 3
    DEFENSE_L6_MAX_CHUNKS_TO_CHECK: int = 5

    # L5: cross-chunk contradiction detection через LLM-judge. Ловит случаи,
    # когда retrieval вернул фрагменты с прямо противоречащими утверждениями —
    # типичная сигнатура гибридного backdoor, в котором target-фраза вшита в
    # тематически легитимный chunk (на уровне отдельного chunk не отличим, но
    # рядом с настоящим правилом виден контраст). +1 LLM-вызов на каждый
    # search_documents tool call с ≥ 2 hits.
    # Действие: 'off' / 'warn' (плашка в ответе) / 'drop' (выкидываем minority
    # chunks по majority-rule: тот файл, у которого больше chunks в выдаче,
    # побеждает).
    DEFENSE_L5_CONTRADICTION_DETECTOR: str = "off"
    # Минимум hits, чтобы L5 запустился. Меньше — не с чем сравнивать.
    DEFENSE_L5_MIN_CHUNKS_TO_CHECK: int = 2
    # Сколько символов из каждого chunk-а скармливать LLM-judge'у. Больше —
    # точнее детект, но дороже по токенам. 500 хватает для большинства фактов.
    DEFENSE_L5_MAX_SNIPPET_CHARS: int = 500

    # Тестовый bypass-режим для security research. Когда включён, в API
    # появляется endpoint /api/security/* для прогонки eval-скриптов
    # (загрузка poisoned-докorpus, сбор метрик). Никогда не включать в prod.
    SECURITY_RESEARCH_MODE: bool = False

    @property
    def data_path(self) -> Path:
        p = Path(self.DATA_DIR).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def uploads_path(self) -> Path:
        p = self.data_path / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_path / "rag.sqlite3"

    @property
    def faiss_path(self) -> Path:
        return self.data_path / "faiss.index"

    @property
    def logs_path(self) -> Path:
        p = self.data_path / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
