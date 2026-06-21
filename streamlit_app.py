"""Streamlit frontend for the Personal Knowledge Assistant."""

from typing import List

import httpx
import streamlit as st

API_URL = "http://localhost:8000/api"


def upload_files(files: List) -> None:
    """Upload PDF files to the backend.

    Args:
        files: List of uploaded file objects from Streamlit.
    """
    for file in files:
        files_data = {"file": (file.name, file.getvalue(), "application/pdf")}
        try:
            response = httpx.post(f"{API_URL}/upload", files=files_data, timeout=120)
            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"✅ Uploaded {result['filename']}: "
                    f"{result['pages_loaded']} pages → {result['chunks_created']} chunks"
                )
            else:
                st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
        except httpx.ConnectError:
            st.error("❌ Cannot connect to backend. Is the API running on port 8000?")
        except Exception as e:
            st.error(f"❌ Upload error: {e}")


def ask_question(question: str, k: int, threshold: float) -> None:
    """Send a question to the chat endpoint.

    Args:
        question: User's question.
        k: Number of chunks to retrieve.
        threshold: Minimum similarity score.
    """
    payload = {"question": question, "k": k, "threshold": threshold}
    try:
        response = httpx.post(f"{API_URL}/chat", json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                }
            )
        else:
            st.error(f"❌ Chat failed: {response.json().get('detail', 'Unknown error')}")
    except httpx.ConnectError:
        st.error("❌ Cannot connect to backend. Is the API running on port 8000?")
    except Exception as e:
        st.error(f"❌ Chat error: {e}")


def fetch_documents() -> List[dict]:
    """Fetch document list from the backend.

    Returns:
        List of document metadata dicts.
    """
    try:
        response = httpx.get(f"{API_URL}/documents", timeout=30)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception:
        pass
    return []


def delete_document(filename: str) -> None:
    """Delete a document via the backend.

    Args:
        filename: Filename to delete.
    """
    try:
        response = httpx.delete(f"{API_URL}/documents/{filename}", timeout=60)
        if response.status_code == 200:
            st.success(f"🗑️ Deleted {filename}")
        else:
            st.error(f"❌ Delete failed: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"❌ Delete error: {e}")


def check_health() -> bool:
    """Check if the backend is healthy.

    Returns:
        True if backend is responding.
    """
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Personal Knowledge Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Personal Knowledge Assistant")
    st.markdown("Upload PDFs and ask questions about your documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("📄 Documents")

        # Health check
        if check_health():
            st.success("🟢 Backend connected")
        else:
            st.error("🔴 Backend disconnected — start FastAPI on port 8000")

        st.divider()

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            if st.button("📤 Upload & Index", type="primary"):
                upload_files(uploaded_files)

        st.divider()

        # Document list
        docs = fetch_documents()
        if docs:
            st.subheader(f"Indexed ({len(docs)} files)")
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    size_kb = doc.get("file_size", 0) / 1024
                    st.text(f"📄 {doc['filename']} ({size_kb:.0f} KB)")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['filename']}"):
                        delete_document(doc["filename"])
                        st.rerun()

            if st.button("🔄 Reindex All"):
                try:
                    response = httpx.post(f"{API_URL}/reindex", timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(
                            f"Reindexed {result['files_indexed']} files, "
                            f"{result['total_chunks']} chunks"
                        )
                except Exception as e:
                    st.error(f"Reindex failed: {e}")
        else:
            st.info("No documents indexed yet.")

        st.divider()

        # Settings
        st.header("⚙️ Settings")
        k = st.slider("Top-K chunks", 1, 20, 5)
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.05)

    # Chat area
    st.divider()

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"- **{src['filename']}** — Page {src['page'] + 1}"
                        )

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Thinking..."):
            ask_question(question, k, threshold)
        st.rerun()

    # Clear chat
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
