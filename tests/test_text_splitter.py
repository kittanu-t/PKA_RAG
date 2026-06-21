"""Tests for text_splitter module."""

import pytest
from langchain_core.documents import Document

from app.rag.text_splitter import get_text_splitter, split_documents


class TestGetTextSplitter:
    def test_default_splitter(self) -> None:
        splitter = get_text_splitter()
        assert splitter is not None

    def test_custom_chunk_size(self) -> None:
        splitter = get_text_splitter(chunk_size=500, chunk_overlap=50)
        assert splitter is not None


class TestSplitDocuments:
    def test_split_creates_chunks(self, sample_documents: list[Document]) -> None:
        chunks = split_documents(sample_documents)
        assert len(chunks) >= len(sample_documents)

    def test_preserves_metadata(self, sample_documents: list[Document]) -> None:
        chunks = split_documents(sample_documents)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert "chunk_id" in chunk.metadata

    def test_chunk_ids_assigned(self, sample_documents: list[Document]) -> None:
        chunks = split_documents(sample_documents)
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]
        assert chunk_ids == list(range(len(chunks)))

    def test_empty_list(self) -> None:
        chunks = split_documents([])
        assert chunks == []
