"""Retriever module — extension point for hybrid search, reranking, etc."""

from typing import List, Optional

from langchain_core.documents import Document

from app.config.settings import settings
from app.rag.vector_store import VectorStoreService
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """Wraps VectorStoreService for document retrieval."""

    def __init__(self, vector_store: VectorStoreService) -> None:
        """Initialize with a vector store instance.

        Args:
            vector_store: VectorStoreService to search against.
        """
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Document]:
        """Retrieve top-k relevant chunks for a query.

        Args:
            query: User's question.
            k: Number of chunks to retrieve. Defaults to settings.top_k.
            threshold: Minimum similarity score. Defaults to settings.similarity_threshold.

        Returns:
            List of relevant Document objects sorted by relevance.
        """
        results = self._vector_store.similarity_search(
            query, k=k or settings.top_k, threshold=threshold
        )
        logger.info("Retriever: found %d chunks for query", len(results))
        return results
