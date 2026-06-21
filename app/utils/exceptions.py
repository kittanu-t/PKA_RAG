"""Custom exceptions for each failure domain."""


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""
    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


class ChatError(Exception):
    """Raised when the chat/LLM pipeline fails."""
    pass
