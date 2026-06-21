# 📚 Personal Knowledge Assistant (RAG)

A production-quality Retrieval-Augmented Generation (RAG) application that lets you upload PDFs and ask questions about your documents. Built with FastAPI, LangChain, Ollama, FAISS, and Streamlit.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   FastAPI (8000) │────▶│  Ollama (11434) │
│  (streamlit_app)│     │   (main.py)      │     │  llama3.2:3b    │
└─────────────────┘     └────────┬─────────┘     │  nomic-embed     │
                                 │               └─────────────────┘
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼────┐ ┌────▼─────┐ ┌───▼──────┐
              │Document  │ │  RAG     │ │ Vector   │
              │Service   │ │ Pipeline │ │ Store    │
              │          │ │          │ │ (FAISS)  │
              │• load    │ │• retrieve│ │          │
              │• split   │ │• prompt  │ │• search  │
              │• store   │ │• chat    │ │• persist │
              └──────────┘ └──────────┘ └──────────┘
```

## Features

- **PDF Upload** — Upload single or multiple PDFs, delete, re-index
- **Document Processing** — Text extraction → cleaning → chunking → embedding → FAISS storage
- **RAG Query Pipeline** — Question → embedding → similarity search → LLM answer with citations
- **Chat Interface** — Conversational Q&A with source citations
- **Anti-Hallucination** — System prompt instructs model to say "I don't know" when context is insufficient
- **Evaluation Module** — Precision@k, Recall@k, faithfulness, context relevance metrics

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) (for local LLM)
- Python 3.12+

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd PKA_RAG

# Install dependencies
uv sync

# Pull Ollama models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Copy environment file
cp .env.example .env

# Start Ollama (if not already running)
ollama serve

# Terminal 1: Start FastAPI backend
uv run python main.py

# Terminal 2: Start Streamlit frontend
uv run streamlit run streamlit_app.py
```

### Docker Setup

```bash
# Build and run (both FastAPI + Streamlit)
docker compose up --build

# Access:
# - FastAPI: http://localhost:8000/docs
# - Streamlit: http://localhost:8501
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `ollama` | API key (use "ollama" for local) |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | LLM endpoint URL |
| `OPENAI_MODEL` | `llama3.2:3b` | LLM model name |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.0` | Minimum similarity score |

## API Examples

### Upload a PDF
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "k": 5}'
```

### List Documents
```bash
curl http://localhost:8000/api/documents
```

### Delete a Document
```bash
curl -X DELETE http://localhost:8000/api/documents/document.pdf
```

### Reindex All
```bash
curl -X POST http://localhost:8000/api/reindex
```

## Testing

```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_document_loader.py -v
```

## Evaluation

```bash
uv run python scripts/evaluate.py
```

Metrics: Precision@k, Recall@k, Context Relevance, Answer Faithfulness.

## Project Structure

```
PKA_RAG/
├── app/
│   ├── api/routes.py          # FastAPI endpoints
│   ├── config/settings.py     # Pydantic Settings
│   ├── models/schemas.py      # Request/response models
│   ├── rag/                   # Core RAG modules
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── prompt_builder.py
│   │   └── chat_service.py
│   ├── services/              # Orchestration layer
│   │   └── document_service.py
│   └── utils/                 # Logging, exceptions
├── data/
│   ├── uploads/               # Uploaded PDFs
│   └── vector_store/          # FAISS index
├── tests/                     # pytest test suite
├── scripts/
│   ├── evaluate.py            # Evaluation metrics
│   └── eval_dataset.json      # Example eval data
├── main.py                    # FastAPI entry point
├── streamlit_app.py           # Streamlit frontend
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Resume Bullets

- Built a Retrieval-Augmented Generation (RAG) application that answers questions from user-uploaded PDFs using LangChain, Ollama Embeddings, and FAISS.
- Implemented document ingestion, vector search, source citation, and conversational memory.
- Designed modular AI pipelines and evaluation metrics for retrieval quality and answer faithfulness.
- Deployed the full stack with Docker Compose, including FastAPI backend and Streamlit frontend.
