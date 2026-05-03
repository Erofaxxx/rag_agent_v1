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
    # консервативный лимит на стороне клиента.
    YANDEX_EMBEDDING_RPM: int = 600
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
