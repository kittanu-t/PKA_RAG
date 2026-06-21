"""Pydantic models for API request/response validation."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    question: str = Field(..., min_length=1, description="User's question")
    k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class SourceInfo(BaseModel):
    """Source citation information."""

    filename: str
    page: int
    chunk_id: int


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str
    sources: List[SourceInfo]


class DocumentInfo(BaseModel):
    """Document metadata."""

    filename: str
    file_size: int
    uploaded: float


class DocumentListResponse(BaseModel):
    """Response model for document listing."""

    documents: List[DocumentInfo]


class UploadResponse(BaseModel):
    """Response model for file upload."""

    filename: str
    pages_loaded: int
    chunks_created: int
    indexed: bool


class DeleteResponse(BaseModel):
    """Response model for document deletion."""

    filename: str
    chunks_deleted: int
    deleted: bool


class ReindexResponse(BaseModel):
    """Response model for reindex operation."""

    files_indexed: int
    total_pages: int
    total_chunks: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    vector_store_empty: bool
    documents_count: int
