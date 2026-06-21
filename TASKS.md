# Task Tracking

## Milestones

### ✅ Milestone 1: Project Scaffolding & Config
- [x] Directory structure
- [x] `pyproject.toml` with all dependencies
- [x] `.env.example`
- [x] `app/config/settings.py` — Pydantic Settings
- [x] `app/utils/logging.py` — logger factory
- [x] `app/utils/exceptions.py` — custom exceptions
- [x] `Dockerfile` + `docker-compose.yml`
- [x] `.gitignore`

### ✅ Milestone 2: Document Ingestion Pipeline
- [x] `app/rag/document_loader.py` — PyMuPDF PDF extraction
- [x] `app/rag/text_splitter.py` — RecursiveCharacterTextSplitter
- [x] `app/rag/embedding_service.py` — Ollama/OpenAI embeddings
- [x] `app/rag/vector_store.py` — FAISS wrapper with persistence
- [x] `app/services/document_service.py` — ingestion orchestration

### ✅ Milestone 3: RAG Query Pipeline
- [x] `app/rag/retriever.py` — search wrapper
- [x] `app/rag/prompt_builder.py` — anti-hallucination prompt
- [x] `app/rag/chat_service.py` — full RAG orchestration

### ✅ Milestone 4: FastAPI Backend
- [x] `app/models/schemas.py` — Pydantic models
- [x] `app/api/routes.py` — all 6 endpoints
- [x] `main.py` — FastAPI app with startup/shutdown

### ✅ Milestone 5: Streamlit Frontend
- [x] `streamlit_app.py` — chat UI with upload, history, citations, settings

### ✅ Milestone 6: Evaluation Module
- [x] `scripts/evaluate.py` — Precision@k, Recall@k, faithfulness, relevance
- [x] `scripts/eval_dataset.json` — example dataset

### ✅ Milestone 7: Testing
- [x] `tests/conftest.py` — shared fixtures
- [x] `tests/test_document_loader.py`
- [x] `tests/test_text_splitter.py`
- [x] `tests/test_vector_store.py`
- [x] `tests/test_retriever.py`
- [x] `tests/test_api.py`
- [x] `tests/test_chat_service.py`

### ✅ Milestone 8: Docker & Documentation
- [x] `Dockerfile` + `docker-compose.yml` + `docker-entrypoint.sh`
- [x] `README.md` — full documentation
- [x] `ARCHITECTURE.md` — detailed architecture
- [x] `CLAUDE.md` — Claude Code guidance
- [x] `TASKS.md` — this file

## Stretch Goals (Optional)

- [ ] Streaming Responses (FastAPI SSE + Streamlit)
- [ ] Conversation Memory (LangChain ConversationBufferMemory)
- [ ] Multi-format Support (TXT, DOCX, Markdown)
- [ ] Hybrid Search (BM25 + Vector with RRF)
- [ ] Reranking (Cross-encoder)
- [ ] Multi-query Retrieval
- [ ] Parent Document Retrieval

## Known Issues / TODOs

- FAISS `delete_by_source` rebuilds the entire index — acceptable for personal scale
- No authentication/authorization — single-user tool
- No rate limiting on API endpoints
- Tests use mocked embeddings — integration tests need Ollama running
