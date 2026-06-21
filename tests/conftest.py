"""Shared test fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from langchain_core.documents import Document


@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    pdf_path = temp_dir / "test_document.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text(fitz.Point(72, 72), "This is page one about machine learning.")

    page2 = doc.new_page()
    page2.insert_text(fitz.Point(72, 72), "This is page two about deep learning and neural networks.")

    page3 = doc.new_page()
    page3.insert_text(fitz.Point(72, 72), "This is page three about natural language processing.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide sample Document objects."""
    return [
        Document(
            page_content="This is page one about machine learning. "
            "Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test_document.pdf", "page": 0},
        ),
        Document(
            page_content="This is page two about deep learning. "
            "Deep learning uses neural networks with many layers.",
            metadata={"source": "test_document.pdf", "page": 1},
        ),
        Document(
            page_content="This is page three about NLP. "
            "Natural language processing deals with text understanding.",
            metadata={"source": "test_document.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def mock_embedding_service():
    """Provide a mock embedding service."""
    mock = MagicMock()
    mock.embed_texts.return_value = [[0.1, 0.2, 0.3]] * 3
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    mock.langchain_embeddings = MagicMock()
    return mock
