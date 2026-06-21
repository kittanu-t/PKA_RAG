"""Tests for retriever module."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.retriever import Retriever


class TestRetriever:
    def test_retrieve_returns_documents(self) -> None:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="Test content", metadata={"source": "test.pdf", "page": 0})
        ]
        retriever = Retriever(mock_vs)
        results = retriever.retrieve("test query")
        assert len(results) == 1
        assert results[0].page_content == "Test content"

    def test_retrieve_empty(self) -> None:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        retriever = Retriever(mock_vs)
        results = retriever.retrieve("unknown topic")
        assert results == []

    def test_retrieve_passes_k_and_threshold(self) -> None:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        retriever = Retriever(mock_vs)
        retriever.retrieve("query", k=10, threshold=0.5)
        mock_vs.similarity_search.assert_called_with("query", k=10, threshold=0.5)
