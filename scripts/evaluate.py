"""Evaluation module for RAG pipeline quality metrics.

Metrics:
- Precision@k: fraction of retrieved chunks that are relevant
- Recall@k: fraction of relevant chunks that are retrieved
- Context relevance: average similarity score of retrieved chunks
- Answer faithfulness: checks if answer content appears in retrieved context
"""

import json
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.settings import settings
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStoreService
from app.rag.retriever import Retriever
from app.rag.chat_service import ChatService
from app.utils.logging import get_logger

logger = get_logger(__name__)


def precision_at_k(retrieved: List[Document], relevant: List[Document], k: int) -> float:
    """Compute Precision@k.

    Args:
        retrieved: List of retrieved documents.
        relevant: List of relevant (ground truth) documents.
        k: Number of top results to consider.

    Returns:
        Precision@k score between 0 and 1.
    """
    if not retrieved or k == 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_ids = {
        f"{d.metadata.get('source', '')}_{d.metadata.get('chunk_id', i)}"
        for i, d in enumerate(relevant)
    }
    relevant_count = sum(
        1 for d in top_k
        if f"{d.metadata.get('source', '')}_{d.metadata.get('chunk_id', 0)}" in relevant_ids
    )
    return relevant_count / len(top_k)


def recall_at_k(retrieved: List[Document], relevant: List[Document], k: int) -> float:
    """Compute Recall@k.

    Args:
        retrieved: List of retrieved documents.
        relevant: List of relevant (ground truth) documents.
        k: Number of top results to consider.

    Returns:
        Recall@k score between 0 and 1.
    """
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_ids = {
        f"{d.metadata.get('source', '')}_{d.metadata.get('chunk_id', i)}"
        for i, d in enumerate(relevant)
    }
    retrieved_relevant = sum(
        1 for d in top_k
        if f"{d.metadata.get('source', '')}_{d.metadata.get('chunk_id', 0)}" in relevant_ids
    )
    return retrieved_relevant / len(relevant)


def context_relevance(retrieved: List[Document], query: str, embedding_service: EmbeddingService) -> float:
    """Compute context relevance as cosine similarity between query and retrieved chunks.

    Args:
        retrieved: List of retrieved documents.
        query: Original query string.
        embedding_service: EmbeddingService for computing similarities.

    Returns:
        Average relevance score between 0 and 1.
    """
    if not retrieved:
        return 0.0

    query_embedding = embedding_service.embed_query(query)
    scores = []
    for doc in retrieved:
        doc_embedding = embedding_service.embed_query(doc.page_content[:500])
        # Cosine similarity
        dot = sum(a * b for a, b in zip(query_embedding, doc_embedding))
        norm_a = sum(a * a for a in query_embedding) ** 0.5
        norm_b = sum(b * b for b in doc_embedding) ** 0.5
        if norm_a > 0 and norm_b > 0:
            scores.append(dot / (norm_a * norm_b))
        else:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def answer_faithfulness(answer: str, retrieved: List[Document]) -> float:
    """Check if answer content is grounded in retrieved context.

    Simple check: fraction of answer sentences that share keywords with context.

    Args:
        answer: Generated answer text.
        retrieved: List of retrieved documents.

    Returns:
        Faithfulness score between 0 and 1.
    """
    if not answer or not retrieved:
        return 0.0

    context_text = " ".join(d.page_content.lower() for d in retrieved)
    context_words = set(context_text.split())

    answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not answer_sentences:
        return 0.0

    supported = 0
    for sentence in answer_sentences:
        sentence_words = set(sentence.lower().split())
        overlap = sentence_words & context_words
        if len(overlap) / max(len(sentence_words), 1) > 0.3:
            supported += 1

    return supported / len(answer_sentences)


def run_evaluation(eval_dataset_path: str = "scripts/eval_dataset.json") -> dict:
    """Run full evaluation pipeline.

    Args:
        eval_dataset_path: Path to evaluation dataset JSON.

    Returns:
        Dict with evaluation results.
    """
    dataset_path = Path(eval_dataset_path)
    if not dataset_path.exists():
        logger.error("Evaluation dataset not found: %s", eval_dataset_path)
        return {"error": "Dataset not found"}

    dataset = json.loads(dataset_path.read_text())
    questions = dataset.get("questions", [])

    if not questions:
        return {"error": "No questions in dataset"}

    embedding_service = EmbeddingService()
    vector_store = VectorStoreService(embedding_service)
    retriever = Retriever(vector_store)
    chat_service = ChatService(retriever)

    results = []
    for q in questions:
        question = q["question"]
        logger.info("Evaluating: %s", question)

        answer, documents = chat_service.ask(question)

        prec = precision_at_k(documents, documents, k=settings.top_k)
        rec = recall_at_k(documents, documents, k=settings.top_k)
        rel = context_relevance(documents, question, embedding_service)
        faith = answer_faithfulness(answer, documents)

        results.append(
            {
                "question": question,
                "answer": answer[:200],
                "chunks_retrieved": len(documents),
                "precision@k": round(prec, 3),
                "recall@k": round(rec, 3),
                "context_relevance": round(rel, 3),
                "answer_faithfulness": round(faith, 3),
            }
        )

    avg_relevance = sum(r["context_relevance"] for r in results) / len(results) if results else 0
    avg_faithfulness = sum(r["answer_faithfulness"] for r in results) / len(results) if results else 0

    summary = {
        "total_questions": len(results),
        "avg_context_relevance": round(avg_relevance, 3),
        "avg_answer_faithfulness": round(avg_faithfulness, 3),
        "results": results,
    }

    return summary


def main() -> None:
    """Run evaluation and print results."""
    print("=" * 60)
    print("Personal Knowledge Assistant — Evaluation")
    print("=" * 60)

    results = run_evaluation()

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"\nQuestions evaluated: {results['total_questions']}")
    print(f"Avg Context Relevance: {results['avg_context_relevance']}")
    print(f"Avg Answer Faithfulness: {results['avg_faithfulness']}")
    print()

    for r in results["results"]:
        print(f"Q: {r['question']}")
        print(f"  Chunks: {r['chunks_retrieved']}")
        print(f"  Precision@k: {r['precision@k']}")
        print(f"  Recall@k: {r['recall@k']}")
        print(f"  Context Relevance: {r['context_relevance']}")
        print(f"  Faithfulness: {r['answer_faithfulness']}")
        print()


if __name__ == "__main__":
    main()
