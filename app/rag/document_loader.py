"""PDF document loader using PyMuPDF."""

from pathlib import Path
from typing import List

import fitz
from langchain_core.documents import Document

from app.utils.exceptions import DocumentLoadError
from app.utils.logging import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF and return one Document per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects with metadata (source, page, file_path).

    Raises:
        DocumentLoadError: If the PDF cannot be loaded.
    """
    try:
        doc = fitz.open(str(file_path))
        documents = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": i,
                            "file_path": str(file_path),
                        },
                    )
                )
        logger.info("Loaded %d pages from %s", len(documents), file_path.name)
        return documents
    except Exception as e:
        raise DocumentLoadError(f"Failed to load {file_path}: {e}") from e


def load_pdfs(file_paths: List[Path]) -> List[Document]:
    """Load multiple PDFs, flattening all pages into a single list.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Combined list of Document objects from all PDFs.
    """
    all_docs: List[Document] = []
    for fp in file_paths:
        all_docs.extend(load_pdf(fp))
    return all_docs
