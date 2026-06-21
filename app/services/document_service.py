"""Document processing pipeline orchestration."""

from pathlib import Path
from typing import List

from app.config.settings import settings
from app.rag.document_loader import load_pdf
from app.rag.text_splitter import split_documents
from app.rag.vector_store import VectorStoreService
from app.rag.embedding_service import EmbeddingService
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Orchestrates the full document ingestion pipeline."""

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize with vector store and embedding service.

        Args:
            vector_store: VectorStoreService for storing embeddings.
            embedding_service: EmbeddingService for generating embeddings.
        """
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._upload_dir: Path = settings.uploads_dir
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def upload_and_index(self, filename: str, content: bytes) -> dict:
        """Save uploaded file → load → split → embed → store.

        Args:
            filename: Original filename.
            content: Raw file bytes.

        Returns:
            Dict with ingestion metadata (filename, pages_loaded, chunks_created, indexed).
        """
        file_path = self._upload_dir / filename
        file_path.write_bytes(content)

        documents = load_pdf(file_path)
        chunks = split_documents(documents)
        self._vector_store.add_documents(chunks)

        result = {
            "filename": filename,
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "indexed": True,
        }
        logger.info("Uploaded and indexed: %s", result)
        return result

    def list_documents(self) -> List[dict]:
        """List all indexed documents with metadata.

        Returns:
            List of dicts with filename, file_size, and uploaded timestamp.
        """
        sources = self._vector_store.list_sources()
        documents = []
        for source in sources:
            file_path = self._upload_dir / source
            stat = file_path.stat() if file_path.exists() else None
            documents.append(
                {
                    "filename": source,
                    "file_size": stat.st_size if stat else 0,
                    "uploaded": stat.st_mtime if stat else 0,
                }
            )
        return documents

    def delete_document(self, filename: str) -> dict:
        """Delete a document from the upload directory and vector store.

        Args:
            filename: Filename to delete.

        Returns:
            Dict with deletion metadata.
        """
        deleted_count = self._vector_store.delete_by_source(filename)

        file_path = self._upload_dir / filename
        if file_path.exists():
            file_path.unlink()

        result = {"filename": filename, "chunks_deleted": deleted_count, "deleted": True}
        logger.info("Deleted document: %s", result)
        return result

    def reindex_all(self) -> dict:
        """Rebuild the entire vector index from uploaded files.

        Returns:
            Dict with reindexing metadata.
        """
        self._vector_store.clear()
        pdf_files = list(self._upload_dir.glob("*.pdf"))

        total_pages = 0
        total_chunks = 0

        for pdf_path in pdf_files:
            documents = load_pdf(pdf_path)
            chunks = split_documents(documents)
            self._vector_store.add_documents(chunks)
            total_pages += len(documents)
            total_chunks += len(chunks)

        result = {
            "files_indexed": len(pdf_files),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
        }
        logger.info("Reindexed all documents: %s", result)
        return result
