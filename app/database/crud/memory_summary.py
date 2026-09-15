from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.embedding import generate_embedding
from app.database.crud.user import read_user

from app.database.model import MemorySummary


def create_memory_summary(
    db: Session,
    user_id: UUID,
    content: str
):

    user = read_user(db, user_id)

    if user is None:
        return None

    memory_summary = MemorySummary(
        user_id = user_id,
        content = content,
        embedding = generate_embedding(content),
    )

    db.add(memory_summary)
    db.commit()
    db.refresh(memory_summary)

    return memory_summary


def read_memory_summary(
    db: Session,
    user_id: UUID,
    memory_summary_id: UUID
):
    statement = select(MemorySummary).where(
        MemorySummary.id == memory_summary_id,
        MemorySummary.user_id == user_id
    )

    memory_summary = db.execute(statement).scalar_one_or_none()

    return memory_summary


def read_memory_summaries(
    db: Session,
    user_id: UUID
):
    statement = (
        select(MemorySummary)
        .where(MemorySummary.user_id == user_id)
    )

    memory_summaries = db.execute(statement).scalars().all()

    return memory_summaries

def read_memory_summaries_by_similarity(
    db: Session,
    user_id: UUID,
    query_embedding: list[float],
    limit: int
):
    distance = MemorySummary.embedding.cosine_distance(query_embedding)

    statement = (
        select(MemorySummary, distance.label("distance"))
        .where(MemorySummary.user_id == user_id)
        .order_by(distance)
        .limit(limit)
    )

    results = db.execute(statement).all()

    return results


def delete_memory_summary(
    db: Session,
    user_id: UUID,
    memory_summary_id: UUID
):
    memory_summary = read_memory_summary(
        db,
        user_id,
        memory_summary_id
    )

    if memory_summary is None:
        return None

    db.delete(memory_summary)
    db.commit()

    return memory_summary

def update_memory_summary(
    db: Session,
    user_id: UUID,
    memory_summary_id: UUID,
    content: str | None = None,
):
    memory_summary = read_memory_summary(
        db,
        user_id,
        memory_summary_id
    )

    if memory_summary is None:
        return None

    if content is not None:
        memory_summary.content = content
        memory_summary.embedding = generate_embedding(content)

    db.commit()
    db.refresh(memory_summary)

    return memory_summary

def update_memory_summary_retrieval_metadata(
    db: Session,
    user_id: UUID,
    memory_summary_id: UUID
):
    memory_summary = read_memory_summary(
        db,
        user_id,
        memory_summary_id
    )

    if memory_summary is None:
        return None

    memory_summary.retrieval_count += 1
    memory_summary.last_retrieved_at = datetime.now()

    db.commit()
    db.refresh(memory_summary)

    return memory_summary