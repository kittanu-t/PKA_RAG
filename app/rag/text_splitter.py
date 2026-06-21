"""Text splitting module using RecursiveCharacterTextSplitter."""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Factory: return a configured RecursiveCharacterTextSplitter.

    Args:
        chunk_size: Max characters per chunk. Defaults to settings.chunk_size.
        chunk_overlap: Overlap between chunks. Defaults to settings.chunk_overlap.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with metadata preservation.

    Preserves original metadata (source, page) and adds chunk_id.

    Args:
        documents: List of Document objects to split.

    Returns:
        List of chunked Document objects with enriched metadata.
    """
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
