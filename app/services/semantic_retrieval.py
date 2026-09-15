from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.model import MemorySummary, MemoryDetail
from app.database.crud.memory_summary import read_memory_summaries_by_similarity
from app.database.crud.memory_detail import read_memory_details_by_similarity, read_memory_details_by_summary

try:
    from app.models.embedding import generate_embedding
except ImportError:
    pass


class PreviousMemory(BaseModel):
    summary_id: UUID
    summary_content: str
    details: list[str]


class RetrievedMemory(BaseModel):
    memory_type: str  # "summary" or "detail"
    memory: MemoryDetail | MemorySummary
    similarity: float

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Context string builders (for future classification)
# ---------------------------------------------------------------------------

def _block_summary_str(summary_blocks: list[dict]) -> str:
    sorted_blocks = sorted(summary_blocks, key=lambda b: b["order"])
    return "\n".join(b["content"] for b in sorted_blocks)


def _recent_message_str(recent_messages: list[dict]) -> str:
    sorted_msgs = sorted(recent_messages, key=lambda m: m["order"])
    lines = []
    for m in sorted_msgs:
        role = m["role"].value if hasattr(m["role"], "value") else str(m["role"])
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _conversation_str(block_str: str, message_str: str, query: str) -> str:
    return (
        f"Previous Conversation Summary: {block_str}\n"
        f"Recent Messages: {message_str}\n"
        f"Query: {query}"
    )


# ---------------------------------------------------------------------------
# Similarity retrieval
# ---------------------------------------------------------------------------

def top_k_retrieval(
    db: Session,
    user_id: UUID,
    query_embedding: list[float],
    k: int,
    cosine_lim: float,
) -> list[RetrievedMemory]:
    k_summary = read_memory_summaries_by_similarity(db, user_id, query_embedding, k)
    k_detail = read_memory_details_by_similarity(db, user_id, query_embedding, k)

    memory: list[RetrievedMemory] = []

    for summary, distance in k_summary:
        similarity = 1 - distance
        if similarity >= cosine_lim:
            memory.append(RetrievedMemory(
                memory_type="summary",
                memory=summary,
                similarity=similarity,
            ))

    for detail, distance in k_detail:
        similarity = 1 - distance
        if similarity >= cosine_lim:
            memory.append(RetrievedMemory(
                memory_type="detail",
                memory=detail,
                similarity=similarity,
            ))

    memory.sort(key=lambda x: x.similarity, reverse=True)
    return memory[:k]


# ---------------------------------------------------------------------------
# Classification (stub — to be implemented later)
# ---------------------------------------------------------------------------

def classification_retrieved_needed(
    retrieved: list[RetrievedMemory],
    conversation_str: str,
) -> list[RetrievedMemory]:
    return retrieved


# ---------------------------------------------------------------------------
# Main entry point — pure retrieval, no dedup, no audit writes
# ---------------------------------------------------------------------------

def semantic_retrieval(
    db: Session,
    user_id: UUID,
    summary_blocks: list[dict],
    recent_messages: list[dict],
    query: str,
) -> list[RetrievedMemory]:
    """
    Retrieve memories relevant to the current query.

    Pure retrieval: embed query, cosine search, classification stub, return.
    No deduplication, no audit writes — caller handles both.

    Args:
        summary_blocks: list of dicts with {order, content}
        recent_messages: list of dicts with {order, role, content}
        query: current user query string
    """
    # 1. Context strings (for future classification)
    block_str = _block_summary_str(summary_blocks)
    message_str = _recent_message_str(recent_messages)
    conversation_str = _conversation_str(block_str, message_str, query)

    # 2. Embed query
    query_embedding = generate_embedding(query)

    # 3. Top-k retrieval
    retrieved = top_k_retrieval(db, user_id, query_embedding, k=5, cosine_lim=0.5)

    # 4. Classification (stub)
    retrieved = classification_retrieved_needed(retrieved, conversation_str)

    return retrieved


# ---------------------------------------------------------------------------
# Ingestion helper — used by ingestion.py
# ---------------------------------------------------------------------------

def retrieve_memory(
    db: Session,
    user_id: UUID,
    embedding: list[float],
    summary_limit: int,
    detail_limit: int,
) -> list[PreviousMemory]:
    """Retrieve top summaries by similarity + their associated details for ingestion."""
    summaries = read_memory_summaries_by_similarity(db, user_id, embedding, summary_limit)
    summaries = [summary for summary, _score in summaries]

    memory: list[PreviousMemory] = []
    for summary in summaries:
        details = read_memory_details_by_summary(db, user_id, summary.id, detail_limit)
        details_content = [d.content for d in details] if details else []
        memory.append(PreviousMemory(
            summary_id=summary.id,
            summary_content=summary.content,
            details=details_content,
        ))

    return memory
