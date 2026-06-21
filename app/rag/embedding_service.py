"""Embedding service using Ollama's native batch embedding API."""

from typing import List

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from app.config.settings import settings
from app.utils.exceptions import EmbeddingError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaBatchEmbeddings(Embeddings):
    """LangChain-compatible Embeddings implementation using Ollama's native batch /api/embed endpoint.

    This avoids the sequential one-at-a-time embedding that langchain_community.OllamaEmbeddings
    performs, which causes timeouts on multi-chunk documents.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        batch_size: int = 32,
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._timeout = timeout

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Send a single batch request to Ollama's native /api/embed endpoint."""
        url = f"{self._base_url}/api/embed"
        payload = {"model": self._model, "input": texts}
        try:
            resp = httpx.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        except httpx.TimeoutException as e:
            raise EmbeddingError(
                f"Ollama embedding request timed out after {self._timeout}s "
                f"for batch of {len(texts)} texts"
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding request failed: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents, processing in batches.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        total = len(texts)
        for i in range(0, total, self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug("Embedding batch %d-%d of %d", i, i + len(batch), total)
            all_embeddings.extend(self._embed_batch(batch))

        logger.info("Embedded %d chunks via Ollama (%d batch(es))", total, (total + self._batch_size - 1) // self._batch_size)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        return self._embed_batch([text])[0]


class EmbeddingService:
    """Wraps Ollama-native batch embeddings for testability and error handling."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the embedding service.

        Args:
            model: Embedding model name. Defaults to settings.embedding_model.

        Raises:
            EmbeddingError: If the embedding client cannot be initialized.
        """
        model = model or settings.embedding_model
        try:
            self._embeddings = OllamaBatchEmbeddings(
                model=model,
                base_url=settings.ollama_base_url,
            )
            logger.info("EmbeddingService initialized with model=%s", model)
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
    def langchain_embeddings(self) -> OllamaBatchEmbeddings:
        """Expose the underlying LangChain embeddings for FAISS."""
        return self._embeddings
