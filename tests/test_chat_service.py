"""Tests for chat_service module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.chat_service import ChatService


class TestChatService:
    def test_ask_returns_answer_and_sources(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"source": "test.pdf", "page": 0, "chunk_id": 0},
            )
        ]

        with patch("app.rag.chat_service.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Machine learning is a subset of artificial intelligence."
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            service = ChatService(mock_retriever)
            answer, docs = service.ask("What is machine learning?")

            assert "machine learning" in answer.lower()
            assert len(docs) == 1

    def test_ask_no_documents(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch("app.rag.chat_service.ChatOpenAI"):
            service = ChatService(mock_retriever)
            answer, docs = service.ask("What is quantum computing?")

            assert "don't know" in answer.lower() or "no relevant" in answer.lower()
            assert docs == []

    def test_ask_multiple_sources(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(
                page_content="Page 1 content about AI.",
                metadata={"source": "doc1.pdf", "page": 0, "chunk_id": 0},
            ),
            Document(
                page_content="Page 2 content about AI.",
                metadata={"source": "doc1.pdf", "page": 1, "chunk_id": 1},
            ),
        ]

        with patch("app.rag.chat_service.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "AI is a broad field. [Source: doc1.pdf, Page 1]"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            service = ChatService(mock_retriever)
            answer, docs = service.ask("Tell me about AI")

            assert len(docs) == 2
