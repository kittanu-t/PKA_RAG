# 📚 Personal Knowledge Assistant (RAG)

> Upload PDFs. Ask questions. Get answers with cited sources.

A production-quality **Retrieval-Augmented Generation (RAG)** application built with Python, FastAPI, LangChain, Ollama, FAISS, and Streamlit. Designed to be portfolio-worthy and resume-ready.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.3-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## ✨ What It Does

1. **Upload** one or more PDF documents
2. The system **processes** them automatically — extracts text, chunks it, generates embeddings, stores in a vector database
3. **Ask questions** about your documents in natural language
4. Get **accurate answers** with **source citations** (filename + page number)
5. The model **refuses to hallucinate** — if the answer isn't in your documents, it says "I don't know"

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   FastAPI (8000) │────▶│  Ollama (11434) │
│   :8501         │     │   :8000          │     │  llama3.2:3b    │
└─────────────────┘     └────────┬─────────┘     │  nomic-embed     │
                                 │               └─────────────────┘
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼────┐ ┌────▼─────┐ ┌───▼──────┐
              │Document  │ │  RAG     │ │ Vector   │
              │Service   │ │ Pipeline │ │ Store    │
              │          │ │          │ │ (FAISS)  │
              │• load    │ │• retrieve│ │• search  │
              │• split   │ │• prompt  │ │• persist │
              │• store   │ │• chat    │ │• delete  │
              └──────────┘ └──────────┘ └──────────┘
```

### Data Flow

**Ingestion (Upload → Index)**
```
PDF file
  → PyMuPDF extracts text per page
  → RecursiveCharacterTextSplitter creates overlapping chunks
  → Ollama nomic-embed-text generates embeddings
  → FAISS stores vectors + metadata (source, page, chunk_id)
  → Persisted to disk (data/vector_store/)
```

**Query (Question → Answer)**
```
User question
  → Embed question with Ollama
  → FAISS similarity search → top-k relevant chunks
  → Build prompt with anti-hallucination instructions + context
  → Ollama llama3.2:3b generates answer
  → Return answer + source citations
```

---

## 🚀 Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) (for local LLM — free)
- Python 3.12+

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/kittanu-t/PKA_RAG.git
cd PKA_RAG

# 2. Install dependencies
uv sync

# 3. Pull Ollama models (llama3.2:3b fits in 4GB VRAM)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 4. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local Ollama)

# 5. Start Ollama (if not running)
ollama serve

# 6. Terminal 1 — Start FastAPI backend
uv run python main.py

# 7. Terminal 2 — Start Streamlit frontend
uv run streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser. Upload a PDF and start asking questions!

### Docker Setup

```bash
# Build and run everything (FastAPI + Streamlit)
docker compose up --build

# Access:
# - Streamlit UI:  http://localhost:8501
# - FastAPI docs:  http://localhost:8000/docs
```

> **Note:** For Docker, set `OPENAI_BASE_URL=http://host.docker.internal:11434/v1` in `.env` so the container can reach Ollama on your host machine.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `ollama` | API key ("ollama" for local, or real key for cloud) |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | LLM endpoint (Ollama, OpenAI, Groq, etc.) |
| `OPENAI_MODEL` | `llama3.2:3b` | LLM model name |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |

**Works with any OpenAI-compatible API** — just change `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Tested with Ollama, OpenAI, and Groq.

---

## 📡 API Reference

All endpoints are prefixed with `/api`. Interactive docs at `http://localhost:8000/docs`.

### Upload a PDF
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@my_document.pdf"
```
```json
{
  "filename": "my_document.pdf",
  "pages_loaded": 25,
  "chunks_created": 48,
  "indexed": true
}
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "k": 5}'
```
```json
{
  "answer": "The main topic is machine learning... [Source: report.pdf, Page 3]",
  "sources": [
    {"filename": "report.pdf", "page": 2, "chunk_id": 5},
    {"filename": "report.pdf", "page": 3, "chunk_id": 0}
  ]
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/documents` | List all uploaded documents |
| `DELETE` | `/api/documents/{filename}` | Remove a document |
| `POST` | `/api/reindex` | Rebuild entire vector index |
| `GET` | `/api/health` | Health check |

---

## 🧪 Testing

```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_document_loader.py -v
```

**27 tests** covering document loading, text splitting, vector store operations, retrieval, API endpoints, and chat service. Current coverage: **72%**.

---

## 📊 Evaluation

Run the evaluation module to measure retrieval and answer quality:

```bash
uv run python scripts/evaluate.py
```

**Metrics:**
- **Precision@k** — Fraction of retrieved chunks that are relevant
- **Recall@k** — Fraction of relevant chunks that were retrieved
- **Context Relevance** — Semantic similarity between query and retrieved chunks
- **Answer Faithfulness** — How grounded the answer is in the retrieved context

---

## 📁 Project Structure

```
PKA_RAG/
├── app/
│   ├── api/
│   │   └── routes.py           # FastAPI REST endpoints
│   ├── config/
│   │   └── settings.py         # Pydantic Settings (env config)
│   ├── models/
│   │   └── schemas.py          # Request/response models
│   ├── rag/                    # Core RAG modules
│   │   ├── document_loader.py  # PDF extraction (PyMuPDF)
│   │   ├── text_splitter.py    # Chunking (RecursiveCharacterTextSplitter)
│   │   ├── embedding_service.py# Ollama/OpenAI embeddings
│   │   ├── vector_store.py     # FAISS CRUD + persistence
│   │   ├── retriever.py        # Search abstraction
│   │   ├── prompt_builder.py   # Anti-hallucination prompts
│   │   └── chat_service.py     # RAG query orchestration
│   ├── services/
│   │   └── document_service.py # Ingestion pipeline orchestration
│   └── utils/
│       ├── logging.py          # Standardized logging
│       └── exceptions.py       # Custom exceptions
├── data/
│   ├── uploads/                # Uploaded PDFs
│   └── vector_store/           # FAISS index files
├── tests/                      # pytest test suite
├── scripts/
│   ├── evaluate.py             # Evaluation metrics
│   └── eval_dataset.json       # Example eval questions
├── main.py                     # FastAPI entry point
├── streamlit_app.py            # Streamlit chat UI
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml              # uv project config
├── .env.example                # Environment template
└── README.md                   # This file
```

---

## 🛡️ Anti-Hallucination Strategy

The system uses multiple techniques to prevent the LLM from making things up:

1. **System prompt** explicitly instructs the model to ONLY use provided context
2. **"I don't know" rule** — model is told to admit when context is insufficient
3. **Low temperature (0.1)** — reduces creative generation, keeps answers factual
4. **Source citations** — model must cite `[Source: filename, Page X]` for every claim
5. **Evaluation module** — measures answer faithfulness automatically

---

## 🎯 Resume Bullets

- Built a Retrieval-Augmented Generation (RAG) application for document Q&A using LangChain, Ollama Embeddings, and FAISS
- Implemented full document ingestion pipeline: PDF extraction → text chunking → embedding → vector storage
- Designed anti-hallucination prompt engineering with source citation requirements
- Created evaluation metrics for retrieval quality (Precision@k, Recall@k) and answer faithfulness
- Deployed full-stack application with FastAPI backend, Streamlit frontend, and Docker Compose
- Wrote comprehensive test suite with 27 tests and 72% code coverage

---

## 🔮 Stretch Goals (Not Yet Implemented)

- [ ] Streaming responses (FastAPI SSE + Streamlit)
- [ ] Conversation memory across messages
- [ ] Multi-format support (TXT, DOCX, Markdown)
- [ ] Hybrid search (BM25 + vector with reciprocal rank fusion)
- [ ] Cross-encoder reranking
- [ ] Multi-query retrieval
- [ ] Parent document retrieval

---

## 📄 License

MIT
