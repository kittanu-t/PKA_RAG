"""Chat service — orchestrates the full RAG query pipeline."""

from typing import List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from app.config.settings import settings
from app.rag.retriever import Retriever
from app.rag.prompt_builder import build_messages
from app.utils.exceptions import ChatError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ChatService:
    """Orchestrates retrieve → prompt → LLM → answer + citations."""

    def __init__(self, retriever: Retriever) -> None:
        """Initialize the chat service.

        Args:
            retriever: Retriever instance for document search.
        """
        self._retriever = retriever
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
            temperature=0.1,
        )
        logger.info("ChatService initialized with model=%s", settings.openai_model)

    def ask(
        self,
        question: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[str, List[Document]]:
        """Full RAG pipeline: retrieve → prompt → LLM → answer.

        Args:
            question: User's question.
            k: Number of chunks to retrieve. Defaults to settings.top_k.
            threshold: Minimum similarity score. Defaults to settings.similarity_threshold.

        Returns:
            Tuple of (answer_text, source_documents).

        Raises:
            ChatError: If the chat pipeline fails.
        """
        try:
            documents = self._retriever.retrieve(question, k=k, threshold=threshold)

            if not documents:
                return (
                    "I don't know. No relevant documents were found in your knowledge base.",
                    [],
                )

            messages = build_messages(question, documents)
            response = self._llm.invoke(messages)
            answer = response.content

            logger.info("Generated answer (%d chars) from %d sources", len(answer), len(documents))
            return answer, documents

        except Exception as e:
            raise ChatError(f"Chat failed: {e}") from e
