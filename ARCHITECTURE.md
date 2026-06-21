# Architecture Documentation

## System Overview

The Personal Knowledge Assistant is a RAG-based document Q&A system. It processes PDF documents into searchable vector embeddings, then uses retrieval-augmented generation to answer user questions with cited sources.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Browser                            │
│  ┌─────────────────────┐    ┌──────────────────────────────┐    │
│  │  Streamlit UI :8501  │    │  Swagger UI :8000/docs       │    │
│  └─────────┬───────────┘    └──────────────┬───────────────┘    │
└────────────┼───────────────────────────────┼─────────────────────┘
             │ HTTP                          │ HTTP
             ▼                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Application :8000                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  app/api/routes.py — REST Endpoints                      │   │
│  │  POST /upload │ POST /chat │ GET /documents │ DELETE │    │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │  Services Layer                                           │   │
│  │  ┌────────────────────┐  ┌──────────────────────────┐   │   │
│  │  │ DocumentService    │  │ ChatService              │   │   │
│  │  │ (orchestration)    │  │ (RAG pipeline)           │   │   │
│  │  └────────┬───────────┘  └────────┬─────────────────┘   │   │
│  └───────────┼───────────────────────┼──────────────────────┘   │
│              │                       │                           │
│  ┌───────────▼───────────────────────▼──────────────────────┐   │
│  │  RAG Core Modules                                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐  │   │
│  │  │ Loader   │→│ Splitter │→│ Embedder  │→│ FAISS     │  │   │
│  │  │(PyMuPDF) │ │(Recurs.) │ │(Ollama)   │ │ Vector DB │  │   │
│  │  └──────────┘ └──────────┘ └───────────┘ └───────────┘  │   │
│  │  ┌──────────┐ ┌──────────────┐ ┌────────────────────┐   │   │
│  │  │ Retriever│ │PromptBuilder │ │ ChatService (LLM)  │   │   │
│  │  └──────────┘ └──────────────┘ └────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Config: app/config/settings.py (Pydantic Settings)      │   │
│  │  Utils:  app/utils/logging.py, app/utils/exceptions.py   │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
             │                               │
             ▼                               ▼
┌──────────────────────┐      ┌──────────────────────────┐
│  Ollama :11434       │      │  File System             │
│  llama3.2:3b (LLM)   │      │  data/uploads/ (PDFs)    │
│  nomic-embed-text    │      │  data/vector_store/      │
└──────────────────────┘      └──────────────────────────┘
```

## Data Flow

### Ingestion Pipeline

```
1. User uploads PDF via Streamlit or API
2. FastAPI receives file bytes → saves to data/uploads/
3. DocumentService.upload_and_index():
   a. document_loader.load_pdf() → List[Document] (one per page)
      - PyMuPDF extracts text per page
      - Metadata: {source, page, file_path}
   b. text_splitter.split_documents() → List[Document] (chunks)
      - RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
      - Metadata enriched with chunk_id
   c. vector_store.add_documents() → FAISS index
      - EmbeddingService generates vectors via Ollama
      - FAISS.from_documents() creates index
      - Index persisted to data/vector_store/
```

### Query Pipeline

```
1. User asks question via Streamlit chat
2. POST /api/chat → ChatRequest {question, k, threshold}
3. ChatService.ask():
   a. Retriever.retrieve(question, k, threshold)
      → VectorStoreService.similarity_search()
      → FAISS similarity_search_with_relevance_scores()
      → Filter by threshold
      → Return top-k Document chunks
   b. PromptBuilder.build_messages(question, documents)
      → System prompt (anti-hallucination rules)
      → Context block with formatted documents
      → User question
   c. ChatOpenAI.invoke(messages) → LLM response
      → Ollama llama3.2:3b generates answer
      → Temperature 0.1 for factual accuracy
4. Return ChatResponse {answer, sources}
   - sources: [{filename, page, chunk_id} for each chunk]
```

## Module Responsibilities

| Module | File | Single Responsibility |
|--------|------|----------------------|
| **Settings** | `app/config/settings.py` | Central configuration from env vars |
| **Logger** | `app/utils/logging.py` | Consistent log formatting |
| **Exceptions** | `app/utils/exceptions.py` | Domain-specific error types |
| **Document Loader** | `app/rag/document_loader.py` | PDF → LangChain Documents |
| **Text Splitter** | `app/rag/text_splitter.py` | Documents → chunks with metadata |
| **Embedding Service** | `app/rag/embedding_service.py` | Text → embedding vectors |
| **Vector Store** | `app/rag/vector_store.py` | FAISS CRUD + persistence |
| **Retriever** | `app/rag/retriever.py` | Search abstraction layer |
| **Prompt Builder** | `app/rag/prompt_builder.py` | Anti-hallucination prompt construction |
| **Chat Service** | `app/rag/chat_service.py` | RAG query orchestration |
| **Document Service** | `app/services/document_service.py` | Ingestion pipeline orchestration |
| **API Routes** | `app/api/routes.py` | HTTP request handling |
| **Schemas** | `app/models/schemas.py` | Request/response validation |
| **Frontend** | `streamlit_app.py` | User interface |

## Design Decisions

### Why Ollama (local LLM)?
- Free, runs on consumer hardware (RTX 3050 4GB VRAM)
- llama3.2:3b fits in 4GB VRAM
- Configurable base URL allows switching to cloud APIs

### Why FAISS?
- File-based, no server needed
- Fast similarity search
- Easy persistence to disk
- Sufficient for personal-scale document collections

### Why RecursiveCharacterTextSplitter?
- LangChain's recommended default
- Tries paragraph → sentence → word → character splits
- Preserves semantic coherence better than fixed-size chunking

### Anti-Hallucination Strategy
1. System prompt explicitly instructs model to only use provided context
2. Model instructed to say "I don't know" when context is insufficient
3. Low temperature (0.1) reduces creative generation
4. Source citations required in responses

## Extension Points

The architecture supports these stretch goals without major refactoring:

1. **Hybrid Search** — Add BM25 retriever alongside vector retriever, merge with reciprocal rank fusion
2. **Reranking** — Add cross-encoder reranker between retriever and prompt builder
3. **Multi-query** — Generate multiple query variations in retriever, merge results
4. **Parent Document** — Store small chunks for search, retrieve parent large chunks for context
5. **Streaming** — Use FastAPI SSE + Streamlit streaming for real-time responses
6. **Conversation Memory** — Add LangChain ConversationBufferMemory to chat_service
7. **Multi-format** — Add TXT, DOCX, Markdown loaders alongside PDF

## Deployment

### Local Development
```bash
ollama serve
uv run python main.py      # Terminal 1
uv run streamlit run streamlit_app.py  # Terminal 2
```

### Docker Production
```bash
docker compose up --build
# FastAPI: http://localhost:8000
# Streamlit: http://localhost:8501
# Data persisted via volume mount: ./data:/app/data
```
