from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    OPENROUTER_API_KEY: str
    LLM_MODEL: str = "deepseek/deepseek-chat"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000
    OPENROUTER_HTTP_REFERER: str = "http://localhost"
    OPENROUTER_X_TITLE: str = "RAG Agent"

    # Auth
    AUTH_USERNAME: str = "admin"
    AUTH_PASSWORD: str = "change_me_please"

    # Paths
    DATA_DIR: str = "./data"

    # Embedding
    # Default — BGE-M3 (~2 GB RAM, лучшее качество, dim=1024).
    # LITE для серверов с 4 GB RAM: EMBEDDING_MODEL=intfloat/multilingual-e5-small
    # (~500 MB RAM, dim=384, чуть хуже на длинных текстах). Для e5-моделей нужны
    # префиксы "passage: " / "query: " — задаются ниже.
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_USE_FP16: bool = True
    EMBEDDING_QUERY_PREFIX: str = ""
    EMBEDDING_PASSAGE_PREFIX: str = ""
    EMBEDDING_MAX_LENGTH: int = 8192

    # Chunking (в символах; русский текст ~3-4 символа на токен)
    CHUNK_SIZE: int = 2400
    CHUNK_OVERLAP: int = 400

    # Search
    SEARCH_TOP_K: int = 7
    SEARCH_USE_BM25: bool = False

    # Limits
    MAX_FILE_SIZE_MB: int = 50
    MAX_DOCUMENTS: int = 100
    MAX_HISTORY_MESSAGES: int = 8

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
