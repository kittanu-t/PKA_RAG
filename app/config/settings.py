"""Application configuration via Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    # LLM
    openai_api_key: str = "ollama"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    similarity_threshold: float = 0.0

    # Paths
    vector_store_path: str = "data/vector_store"
    upload_dir: str = "data/uploads"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def vector_store_dir(self) -> Path:
        return Path(self.vector_store_path)

    @property
    def uploads_dir(self) -> Path:
        return Path(self.upload_dir)


settings = Settings()
