"""FastAPI application entry point."""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config.settings import settings
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStoreService
from app.rag.retriever import Retriever
from app.rag.chat_service import ChatService
from app.services.document_service import DocumentService
from app.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    logger.info("Starting up Personal Knowledge Assistant...")
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService(embedding_service)
    retriever = Retriever(vector_store)
    chat_service = ChatService(retriever)
    document_service = DocumentService(vector_store, embedding_service)

    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.chat_service = chat_service
    app.state.document_service = document_service
    logger.info("All services initialized successfully")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Personal Knowledge Assistant",
        description="RAG-based document Q&A system",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api", tags=["api"])
    return app


app = create_app()


def main() -> None:
    """Run the application."""
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
