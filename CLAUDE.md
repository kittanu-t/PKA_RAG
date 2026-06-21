# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working on this repository.

## Project Overview

**Personal Knowledge Assistant (RAG)** — A document Q&A system that lets users upload PDFs and ask questions using Retrieval-Augmented Generation. Built with FastAPI, LangChain, Ollama (local LLM), FAISS, and Streamlit.

## Tech Stack

- **Python 3.12** with **uv** package manager
- **FastAPI** backend (port 8000)
- **Streamlit** frontend (port 8501)
- **LangChain** RAG framework
- **Ollama** local LLM (`llama3.2:3b`) — configurable via `.env`
- **FAISS** vector database (file-based, persisted to `data/vector_store/`)
- **PyMuPDF** for PDF text extraction
- **Docker** support via `docker compose up`

## Common Commands

```bash
# Install dependencies
uv sync

# Run FastAPI backend only
uv run python main.py

# Run Streamlit frontend only (in another terminal)
uv run streamlit run streamlit_app.py

# Run tests
uv run pytest tests/ -v --cov=app --cov-report=term-missing

# Run evaluation
uv run python scripts/evaluate.py

# Docker (runs both FastAPI + Streamlit)
docker compose up --build
```

## Architecture

### Data Flow — Ingestion
```
PDF upload → document_loader (PyMuPDF → LangChain Documents)
           → text_splitter (RecursiveCharacterTextSplitter → chunks)
           → vector_store (FAISS.from_documents → embed + store)
           → persist to data/vector_store/
```

### Data Flow — Query
```
User question → retriever (FAISS similarity_search → top-k chunks)
             → prompt_builder (anti-hallucination prompt + context)
             → chat_service (Ollama LLM → answer + citations)
```

### Key Modules

| Module | File | Purpose |
|--------|------|---------|
| Config | `app/config/settings.py` | Pydantic Settings singleton from `.env` |
| Document Loader | `app/rag/document_loader.py` | PyMuPDF PDF extraction → Documents |
| Text Splitter | `app/rag/text_splitter.py` | RecursiveCharacterTextSplitter |
| Embedding Service | `app/rag/embedding_service.py` | Wraps Ollama/OpenAI embeddings |
| Vector Store | `app/rag/vector_store.py` | FAISS wrapper with persistence |
| Retriever | `app/rag/retriever.py` | Search wrapper (extension point) |
| Prompt Builder | `app/rag/prompt_builder.py` | Anti-hallucination prompt + citations |
| Chat Service | `app/rag/chat_service.py` | Full RAG query orchestration |
| Document Service | `app/services/document_service.py` | Upload/index/delete/reindex pipeline |
| API Routes | `app/api/routes.py` | FastAPI endpoints |
| Schemas | `app/models/schemas.py` | Pydantic request/response models |
| Frontend | `streamlit_app.py` | Streamlit chat UI |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload PDF file(s) |
| POST | `/api/chat` | Ask a question, get answer + sources |
| GET | `/api/documents` | List uploaded documents |
| DELETE | `/api/documents/{filename}` | Delete a document |
| POST | `/api/reindex` | Rebuild entire vector index |
| GET | `/api/health` | Health check |

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

The `OPENAI_BASE_URL` and `OPENAI_API_KEY` fields work with any OpenAI-compatible endpoint (Ollama, Groq, OpenRouter, etc.).

## Design Patterns

- **Dependency Injection**: Services accept dependencies as constructor params for testability
- **Pydantic Settings**: All config flows through a central `settings` singleton
- **Custom Exceptions**: `DocumentLoadError`, `EmbeddingError`, `VectorStoreError`, `ChatError`
- **Modular Pipeline**: Each RAG component is independently testable
- **Stateless API, Stateful Vector DB**: FastAPI is stateless; state lives in FAISS on disk

## Testing

Tests are in `tests/` using pytest. Fixtures in `conftest.py` provide sample PDFs and mock embedding services. Target: >80% coverage.

```bash
# Run all tests
uv run pytest tests/ -v

# Run single test file
uv run pytest tests/test_document_loader.py -v

# Run with coverage
uv run pytest tests/ -v --cov=app --cov-report=term-missing
```

## Stretch Goals (not yet implemented)

1. Streaming Responses (SSE)
2. Conversation Memory (LangChain ConversationBufferMemory)
3. Multi-format Support (TXT, DOCX, Markdown)
4. Hybrid Search (BM25 + Vector)
5. Reranking (Cross-encoder)
6. Multi-query Retrieval
7. Parent Document Retrieval
