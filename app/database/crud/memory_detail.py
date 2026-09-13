from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import MemoryDetail
from app.models.embedding import generate_embedding

from app.database.crud.chat import read_chat
from app.database.crud.memory_summary import read_memory_summary


def create_memory_detail(
    db: Session,
    user_id: UUID,
    summary_id: UUID,
    chat_id: UUID,
    content: str
):
    summary = read_memory_summary(
        db,
        user_id,
        summary_id
    )

    if summary is None:
        return None

    chat = read_chat(
        db,
        user_id,
        chat_id
    )

    if chat is None:
        return None

    memory_detail = MemoryDetail(
        user_id=user_id,
        summary_id=summary_id,
        chat_id=chat_id,
        content=content,
        embedding=generate_embedding(content)
    )

    db.add(memory_detail)
    db.commit()
    db.refresh(memory_detail)

    return memory_detail

def read_memory_detail(
    db: Session,
    user_id: UUID,
    memory_detail_id: UUID
):
    statement = select(MemoryDetail).where(
        MemoryDetail.id == memory_detail_id,
        MemoryDetail.user_id == user_id
    )

    memory_detail = db.execute(
        statement
    ).scalar_one_or_none()

    return memory_detail

def read_memory_details_by_similarity(
    db: Session,
    user_id: UUID,
    query_embedding: list[float],
    limit: int
):
    distance = MemoryDetail.embedding.cosine_distance(query_embedding)

    statement = (
        select(MemoryDetail, distance.label("distance"))
        .where(MemoryDetail.user_id == user_id)
        .order_by(distance)
        .limit(limit)
    )

    results = db.execute(statement).all()

    return results

def read_memory_details_by_summary(
    db: Session,
    user_id: UUID,
    summary_id: UUID,
    limit: int
):
    summary = read_memory_summary(
        db,
        user_id,
        summary_id
    )

    if summary is None:
        return None

    statement = (
        select(MemoryDetail)
        .where(
            MemoryDetail.summary_id == summary_id,
            MemoryDetail.user_id == user_id
        )
        .order_by(MemoryDetail.created_at.desc())
        .limit(limit)
    )

    memory_details = db.execute(statement).scalars().all()

    return memory_details


def read_memory_details_by_chat(
    db: Session,
    user_id: UUID,
    chat_id: UUID
):
    chat = read_chat(
        db,
        user_id,
        chat_id
    )

    if chat is None:
        return None

    statement = (
        select(MemoryDetail)
        .where(
            MemoryDetail.chat_id == chat_id)
        .order_by(MemoryDetail.created_at)
    )

    memory_details = db.execute(
        statement
    ).scalars().all()

    return memory_details


def delete_memory_detail(
    db: Session,
    user_id: UUID,
    memory_detail_id: UUID
):
    memory_detail = read_memory_detail(
        db,
        user_id,
        memory_detail_id
    )

    if memory_detail is None:
        return None

    db.delete(memory_detail)
    db.commit()

    return memory_detail

