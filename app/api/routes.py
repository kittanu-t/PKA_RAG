"""FastAPI route definitions."""

from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    ReindexResponse,
    SourceInfo,
    UploadResponse,
)
from app.services.document_service import DocumentService
from app.rag.chat_service import ChatService

router = APIRouter()


def get_document_service() -> DocumentService:
    """Get the document service from app state."""
    from main import app
    return app.state.document_service


def get_chat_service() -> ChatService:
    """Get the chat service from app state."""
    from main import app
    return app.state.chat_service


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and index a PDF file.

    Args:
        file: PDF file to upload.

    Returns:
        UploadResponse with ingestion metadata.

    Raises:
        HTTPException: If the file is not a PDF or processing fails.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        doc_service = get_document_service()
        result = doc_service.upload_and_index(file.filename, content)
        return UploadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}") from e


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Ask a question about uploaded documents.

    Args:
        request: ChatRequest with question and optional parameters.

    Returns:
        ChatResponse with answer and source citations.

    Raises:
        HTTPException: If the chat pipeline fails.
    """
    try:
        chat_service = get_chat_service()
        answer, documents = chat_service.ask(
            question=request.question,
            k=request.k,
            threshold=request.threshold,
        )

        sources = [
            SourceInfo(
                filename=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", 0),
                chunk_id=doc.metadata.get("chunk_id", 0),
            )
            for doc in documents
        ]

        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}") from e


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all uploaded and indexed documents.

    Returns:
        DocumentListResponse with list of document metadata.
    """
    try:
        doc_service = get_document_service()
        docs = doc_service.list_documents()
        return DocumentListResponse(
            documents=[
                {
                    "filename": d["filename"],
                    "file_size": d["file_size"],
                    "uploaded": d["uploaded"],
                }
                for d in docs
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}") from e


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str) -> DeleteResponse:
    """Delete a document and its indexed chunks.

    Args:
        filename: Name of the file to delete.

    Returns:
        DeleteResponse with deletion metadata.

    Raises:
        HTTPException: If deletion fails.
    """
    try:
        doc_service = get_document_service()
        result = doc_service.delete_document(filename)
        return DeleteResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}") from e


@router.post("/reindex", response_model=ReindexResponse)
async def reindex() -> ReindexResponse:
    """Rebuild the entire vector index from uploaded PDFs.

    Returns:
        ReindexResponse with reindexing metadata.

    Raises:
        HTTPException: If reindexing fails.
    """
    try:
        doc_service = get_document_service()
        result = doc_service.reindex_all()
        return ReindexResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}") from e


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with system status.
    """
    from main import app
    doc_service = app.state.document_service
    sources = doc_service._vector_store.list_sources()
    return HealthResponse(
        status="healthy",
        vector_store_empty=doc_service._vector_store.is_empty,
        documents_count=len(sources),
    )
