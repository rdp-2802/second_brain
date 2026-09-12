from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import KnowledgeDetail
from app.models.embedding import generate_embedding


from app.database.crud.chat import get_chat
from app.database.crud.knowledge_summary import (
    get_knowledge_summary
)


def add_knowledge_detail(
    db: Session,
    summary_id: UUID,
    chat_id: UUID,
    source_message_ids: list[UUID],
    detail_content: str,
    user_id: UUID
):
    knowledge_summary = get_knowledge_summary(
        db,
        summary_id
    )

    if knowledge_summary is None:
        return None

    chat = get_chat(db, chat_id)

    if chat is None:
        return None

    knowledge_detail = KnowledgeDetail(
        user_id = user_id
        summary_id=summary_id,
        chat_id=chat_id,
        source_message_ids=source_message_ids,
        detail_content=detail_content,
        embedding=generate_embedding(detail_content)
    )

    db.add(knowledge_detail)
    db.commit()
    db.refresh(knowledge_detail)

    return knowledge_detail


def get_knowledge_detail(
    db: Session,
    knowledge_detail_id: UUID
):
    statement = select(KnowledgeDetail).where(
        KnowledgeDetail.id == knowledge_detail_id
    )

    knowledge_detail = db.execute(
        statement
    ).scalar_one_or_none()

    return knowledge_detail

def get_near_embeddings(db: Session, user_id: UUID, limit: int, query_embedding: List[int]):
    distance = KnowledgeDetail.embedding.cosine_distance(query_embedding)

    statement = select(KnowledgeDetail, distance.label("distance")).where(KnowledgeDetail.user_id == user_id).order_by(KnowledgeDetail.embedding.cosine_distance(query_embedding)).limit(limit)
    result = db.execute(statement).all()
    return result

def get_knowledge_details_by_summary(
    db: Session,
    summary_id: UUID,
    limit: int
):
    statement = select(KnowledgeDetail).where(
        KnowledgeDetail.summary_id == summary_id
    ).order_by(KnowledgeDetail.created_at.desc()).limit(limit)

    knowledge_details = db.execute(
        statement
    ).scalars().all()

    return knowledge_details


def get_knowledge_details_by_chat(
    db: Session,
    chat_id: UUID,
    user_id: UUID
):
    statement = select(KnowledgeDetail).where(
        KnowledgeDetail.chat_id == chat_id & KnowledgeDetail.user_id == user_id
    )

    knowledge_details = db.execute(
        statement
    ).scalars().all()

    return knowledge_details


def delete_knowledge_detail(
    db: Session,
    knowledge_detail_id: UUID
):
    knowledge_detail = get_knowledge_detail(
        db,
        knowledge_detail_id
    )

    if knowledge_detail is None:
        return None

    db.delete(knowledge_detail)
    db.commit()

    return knowledge_detail

#never to be used
def update_knowledge_detail(
    db: Session,
    knowledge_detail_id: UUID,
    detail_content: str | None = None,
):
    knowledge_detail = get_knowledge_detail(
        db,
        knowledge_detail_id
    )

    if knowledge_detail is None:
        return None

    if detail_content is not None:
        knowledge_detail.detail_content = detail_content
        knowledge_detail.embedding=generate_embedding(detail_content)


    db.commit()
    db.refresh(knowledge_detail)

    return knowledge_detail