"""Embedding service wrapping OpenAI/Ollama embeddings."""

from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config.settings import settings
from app.utils.exceptions import EmbeddingError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Wraps OpenAI-compatible embeddings for testability and error handling."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the embedding service.

        Args:
            model: Embedding model name. Defaults to settings.embedding_model.

        Raises:
            EmbeddingError: If the embedding client cannot be initialized.
        """
        try:
            self._embeddings = OpenAIEmbeddings(
                model=model or settings.embedding_model,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_base_url,
            )
            logger.info("EmbeddingService initialized with model=%s", model or settings.embedding_model)
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embeddings: {e}") from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        return self._embeddings.embed_query(text)

    @property
    def langchain_embeddings(self) -> OpenAIEmbeddings:
        """Expose the underlying LangChain embeddings for FAISS."""
        return self._embeddings
