"""Tests for document_loader module."""

from pathlib import Path

import fitz
import pytest
from langchain_core.documents import Document

from app.rag.document_loader import load_pdf, load_pdfs
from app.utils.exceptions import DocumentLoadError


class TestLoadPdf:
    def test_load_valid_pdf(self, sample_pdf: Path) -> None:
        docs = load_pdf(sample_pdf)
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_populated(self, sample_pdf: Path) -> None:
        docs = load_pdf(sample_pdf)
        for doc in docs:
            assert doc.metadata["source"] == "test_document.pdf"
            assert "file_path" in doc.metadata
            assert "page" in doc.metadata

    def test_page_numbers(self, sample_pdf: Path) -> None:
        docs = load_pdf(sample_pdf)
        assert docs[0].metadata["page"] == 0
        assert docs[1].metadata["page"] == 1
        assert docs[2].metadata["page"] == 2

    def test_text_content(self, sample_pdf: Path) -> None:
        docs = load_pdf(sample_pdf)
        assert "page one" in docs[0].page_content
        assert "page two" in docs[1].page_content

    def test_nonexistent_file(self) -> None:
        with pytest.raises(DocumentLoadError):
            load_pdf(Path("/nonexistent/file.pdf"))


class TestLoadPdfs:
    def test_load_multiple(self, sample_pdf: Path, temp_dir: Path) -> None:
        pdf2 = temp_dir / "second.pdf"
        doc = fitz.open()
        doc.new_page().insert_text(fitz.Point(72, 72), "Second PDF content.")
        doc.save(str(pdf2))
        doc.close()

        docs = load_pdfs([sample_pdf, pdf2])
        assert len(docs) == 4

    def test_empty_list(self) -> None:
        docs = load_pdfs([])
        assert docs == []
