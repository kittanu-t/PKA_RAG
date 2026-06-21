"""Prompt builder with anti-hallucination instructions and source citation."""

from typing import List

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a Personal Knowledge Assistant. Answer questions based ONLY on the provided context documents.

Rules:
1. Answer ONLY using information from the provided context.
2. If the context does not contain enough information to answer the question, say "I don't know. The provided documents don't contain information about this topic."
3. Do NOT hallucinate or make up information.
4. Always cite your sources using the format [Source: filename, Page X].
5. Be concise but thorough.
6. If multiple sources support your answer, cite all relevant ones."""


def format_context(documents: List[Document]) -> str:
    """Format retrieved documents into a context string.

    Args:
        documents: List of retrieved Document objects.

    Returns:
        Formatted context string with source annotations.
    """
    parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Doc {i+1}] Source: {source}, Page {page}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_prompt(query: str, documents: List[Document]) -> str:
    """Build a RAG prompt from query and retrieved documents.

    Args:
        query: User's question.
        documents: List of retrieved Document objects.

    Returns:
        Formatted prompt string.
    """
    context_text = format_context(documents)
    return f"""{SYSTEM_PROMPT}

--- Context Documents ---
{context_text}
--- End Context ---

Question: {query}

Answer:"""


def build_messages(query: str, documents: List[Document]) -> List[dict]:
    """Build a messages list for OpenAI ChatCompletion API.

    Args:
        query: User's question.
        documents: List of retrieved Document objects.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    context_text = format_context(documents)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]
