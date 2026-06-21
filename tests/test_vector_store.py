"""Tests for vector_store module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.vector_store import VectorStoreService


class TestVectorStoreService:
    def test_add_documents(self, temp_dir, mock_embedding_service, sample_documents) -> None:
        with patch("app.rag.vector_store.settings") as mock_settings:
            mock_settings.vector_store_path = str(temp_dir / "vs")
            mock_settings.top_k = 5
            mock_settings.similarity_threshold = 0.0

            from langchain_community.vectorstores import FAISS as RealFAISS
            with patch.object(RealFAISS, "from_documents") as mock_from_docs:
                mock_db = MagicMock()
                mock_db.index.ntotal = 3
                mock_from_docs.return_value = mock_db

                service = VectorStoreService(mock_embedding_service)
                ids = service.add_documents(sample_documents)
                assert len(ids) == 3

    def test_list_sources_empty(self, temp_dir, mock_embedding_service) -> None:
        with patch("app.rag.vector_store.settings") as mock_settings:
            mock_settings.vector_store_path = str(temp_dir / "vs2")
            mock_settings.top_k = 5
            mock_settings.similarity_threshold = 0.0

            service = VectorStoreService(mock_embedding_service)
            sources = service.list_sources()
            assert sources == []

    def test_clear(self, temp_dir, mock_embedding_service) -> None:
        with patch("app.rag.vector_store.settings") as mock_settings:
            mock_settings.vector_store_path = str(temp_dir / "vs3")
            mock_settings.top_k = 5
            mock_settings.similarity_threshold = 0.0

            service = VectorStoreService(mock_embedding_service)
            service.clear()
            assert service.is_empty
