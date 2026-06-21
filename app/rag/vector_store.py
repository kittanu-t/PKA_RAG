"""FAISS vector store service with persistence."""

import json
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config.settings import settings
from app.rag.embedding_service import EmbeddingService
from app.utils.exceptions import VectorStoreError
from app.utils.logging import get_logger

logger = get_logger(__name__)

METADATA_FILE = "index_metadata.json"


class VectorStoreService:
    """FAISS-based vector store with add, search, delete, and persistence."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """Initialize the vector store, loading from disk if available.

        Args:
            embedding_service: EmbeddingService instance for generating vectors.
        """
        self._embedding_service = embedding_service
        self._store_dir: Path = settings.vector_store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._db: Optional[FAISS] = None
        self._metadata: dict = {}
        self._load()

    def _load(self) -> None:
        """Load existing FAISS index from disk, or initialize empty."""
        index_path = self._store_dir / "index.faiss"
        if index_path.exists():
            try:
                self._db = FAISS.load_local(
                    str(self._store_dir),
                    self._embedding_service.langchain_embeddings,
                    allow_dangerous_deserialization=True,
                )
                meta_path = self._store_dir / METADATA_FILE
                if meta_path.exists():
                    self._metadata = json.loads(meta_path.read_text())
                logger.info("Loaded FAISS index with %d vectors", self._db.index.ntotal)
            except Exception as e:
                logger.warning("Failed to load index, creating new: %s", e)
                self._db = None
                self._metadata = {}

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to embed and store.

        Returns:
            List of document IDs generated for the added documents.

        Raises:
            VectorStoreError: If documents cannot be added.
        """
        if not documents:
            return []

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for doc in documents:
            doc.metadata["upload_timestamp"] = timestamp

        try:
            if self._db is None:
                self._db = FAISS.from_documents(
                    documents,
                    self._embedding_service.langchain_embeddings,
                )
            else:
                self._db.add_documents(documents)

            self._persist()
            ids = [
                f"{d.metadata.get('source', 'unknown')}_{d.metadata.get('chunk_id', i)}"
                for i, d in enumerate(documents)
            ]
            logger.info("Added %d chunks to vector store", len(documents))
            return ids
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> List[Document]:
        """Search for top-k similar chunks.

        Args:
            query: Search query string.
            k: Number of results. Defaults to settings.top_k.
            threshold: Minimum similarity score. Defaults to settings.similarity_threshold.

        Returns:
            List of relevant Document objects sorted by relevance.
        """
        k = k or settings.top_k
        threshold = threshold if threshold is not None else settings.similarity_threshold

        if self._db is None:
            return []

        results = self._db.similarity_search_with_relevance_scores(query, k=k)
        filtered = [(doc, score) for doc, score in results if score >= threshold]
        logger.info("Retrieved %d chunks (k=%d, threshold=%.2f)", len(filtered), k, threshold)
        return [doc for doc, _ in filtered]

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source file by rebuilding the index.

        Args:
            source: Filename to delete.

        Returns:
            Number of chunks deleted.
        """
        if self._db is None:
            return 0

        all_docs = self._get_all_documents()
        remaining = [d for d in all_docs if d.metadata.get("source") != source]
        deleted_count = len(all_docs) - len(remaining)

        if remaining:
            self._db = FAISS.from_documents(
                remaining,
                self._embedding_service.langchain_embeddings,
            )
        else:
            self._db = None

        self._persist()
        logger.info("Deleted %d chunks for source=%s", deleted_count, source)
        return deleted_count

    def list_sources(self) -> List[str]:
        """List all unique source filenames in the store.

        Returns:
            List of unique source filenames.
        """
        if self._db is None:
            return []
        docs = self._get_all_documents()
        return list({d.metadata.get("source", "unknown") for d in docs})

    def clear(self) -> None:
        """Clear the entire vector store."""
        self._db = None
        self._metadata = {}
        self._persist()
        logger.info("Cleared vector store")

    @property
    def is_empty(self) -> bool:
        """Check if the vector store has no documents."""
        return self._db is None

    def _persist(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self._db is not None:
            self._db.save_local(str(self._store_dir))
        meta_path = self._store_dir / METADATA_FILE
        meta_path.write_text(json.dumps(self._metadata))

    def _get_all_documents(self) -> List[Document]:
        """Retrieve all documents from the FAISS docstore.

        Returns:
            List of all stored Document objects.
        """
        if self._db is None:
            return []
        return [
            self._db.docstore.search(doc_id)
            for doc_id in self._db.docstore._dict.keys()
        ]
