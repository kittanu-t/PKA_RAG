"""Tests for API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_services():
    """Set up mock services on the app state."""
    mock_doc_service = MagicMock()
    mock_chat_service = MagicMock()
    mock_vector_store = MagicMock()

    mock_doc_service._vector_store = mock_vector_store
    mock_vector_store.list_sources.return_value = []

    return mock_doc_service, mock_chat_service


@pytest.fixture
def client(mock_services):
    """Create a test client with mocked services."""
    mock_doc_service, mock_chat_service = mock_services

    with patch("app.api.routes.get_document_service", return_value=mock_doc_service), \
         patch("app.api.routes.get_chat_service", return_value=mock_chat_service):
        from main import app
        app.state.document_service = mock_doc_service
        app.state.chat_service = mock_chat_service
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client: TestClient, mock_services) -> None:
        _, _ = mock_services
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestChatEndpoint:
    def test_chat(self, client: TestClient, mock_services) -> None:
        _, mock_chat_service = mock_services
        mock_chat_service.ask.return_value = (
            "Machine learning is a subset of AI.",
            [
                MagicMock(
                    metadata={"source": "test.pdf", "page": 0, "chunk_id": 0},
                    page_content="ML content",
                )
            ],
        )

        response = client.post("/api/chat", json={"question": "What is ML?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_chat_empty_question(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"question": ""})
        assert response.status_code == 422


class TestDocumentsEndpoint:
    def test_list_documents(self, client: TestClient, mock_services) -> None:
        mock_doc_service, _ = mock_services
        mock_doc_service.list_documents.return_value = [
            {"filename": "test.pdf", "file_size": 1024, "uploaded": 1234567890}
        ]

        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 1

    def test_delete_document(self, client: TestClient, mock_services) -> None:
        mock_doc_service, _ = mock_services
        mock_doc_service.delete_document.return_value = {
            "filename": "test.pdf",
            "chunks_deleted": 5,
            "deleted": True,
        }

        response = client.delete("/api/documents/test.pdf")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
