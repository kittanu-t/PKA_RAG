#!/bin/bash
set -e

# Start FastAPI in background
uv run uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
exec uv run streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
