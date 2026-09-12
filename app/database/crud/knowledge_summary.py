from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.embedding import generate_embedding

from app.database.model import (
    KnowledgeSummary,
)

#relevancy score is set to be 0 right now. both while adding and updating a summary. we will develop a useful algorithm in v2 or v3 for calculating and using relevancy score

def add_knowledge_summary(
    db: Session,
    summary: str,
    user_id: UUID
):
    knowledge_summary = KnowledgeSummary(
        user_id=user_id,
        summary=summary,
        embedding=generate_embedding(summary),
        relevancy_score=0
    )

    db.add(knowledge_summary)
    db.commit()
    db.refresh(knowledge_summary)

    return knowledge_summary


def get_knowledge_summary(
    db: Session,
    knowledge_summary_id: UUID,
    user_id: UUID
):
    statement = select(KnowledgeSummary).where(
        KnowledgeSummary.id == knowledge_summary_id & Knowledge.user_id == user_id
    )

    knowledge_summary = db.execute(
        statement
    ).scalar_one_or_none()

    return knowledge_summary


def get_all_knowledge_summaries(db: Session):
    statement = select(KnowledgeSummary)

    knowledge_summaries = db.execute(
        statement
    ).scalars().all()

    return knowledge_summaries

def get_near_embeddings(db: Session, user_id: UUID, limit: int, query_embedding: List[int]):
    distance = KnowledgeSummary.embedding.cosine_distance(query_embedding)

    statement = select(KnowledgeSummary, distance.label("distance")).where(KnowledgeSummary.user_id == user_id).order_by(KnowledgeSummary.embedding.cosine_distance(query_embedding)).limit(limit)
    results = db.execute(statement).all()
    
    for summary, distance in results:
        summary.retrieval_count += 1
        summary.last_retrieved_at = datetime.now()

    db.commit()

    return results


def delete_knowledge_summary(
    db: Session,
    knowledge_summary_id: UUID
):
    knowledge_summary = get_knowledge_summary(
        db,
        knowledge_summary_id
    )

    if knowledge_summary is None:
        return None

    db.delete(knowledge_summary)
    db.commit()

    return knowledge_summary


def update_knowledge_summary(
    db: Session,
    user_id: UUID,
    knowledge_summary_id: UUID,
    summary: str | None = None,
):
    knowledge_summary = get_knowledge_summary(
        db,
        knowledge_summary_id,
        user_id
    )

    if knowledge_summary is None:
        return None

    if summary is not None:
        knowledge_summary.summary = summary
        knowledge_summary.updated_at = datetime.now()
        knowledge_summary.embedding = generate_embedding(summary)


    db.commit()
    db.refresh(knowledge_summary)

    return knowledge_summary

def update_knowsum_last_retrieve_time(db: Session, id: UUID):
    statement = select(KnowledgeSummary).where(KnowledgeSummary.id == id)
    result = db.execute(statement).scalar_one_or_none()
    result.last_retrieved_at = datetime.now()
    db.commit()
    db.refresh(result)

    return result