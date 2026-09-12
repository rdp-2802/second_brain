from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import ChatRetrievedSummary


def add_chat_retrieved_summary(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID,
    knowledge_summary_id: UUID
):
    retrieved_summary = ChatRetrievedSummary(
        user_id=user_id,
        chat_id=chat_id,
        message_id=message_id,
        knowledge_summary_id=knowledge_summary_id
    )

    db.add(retrieved_summary)
    db.commit()
    db.refresh(retrieved_summary)

    return retrieved_summary


def get_chat_retrieved_summary(
    db: Session,
    retrieved_summary_id: UUID
):
    statement = select(ChatRetrievedSummary).where(
        ChatRetrievedSummary.id == retrieved_summary_id
    )

    retrieved_summary = db.execute(statement).scalar_one_or_none()

    return retrieved_summary


def get_chat_retrieved_summaries_by_chat(
    db: Session,
    chat_id: UUID
):
    statement = (
        select(ChatRetrievedSummary)
        .where(ChatRetrievedSummary.chat_id == chat_id)
    )

    retrieved_summaries = db.execute(statement).scalars().all()

    return retrieved_summaries


def get_chat_retrieved_summaries_by_message(
    db: Session,
    message_id: UUID
):
    statement = (
        select(ChatRetrievedSummary)
        .where(ChatRetrievedSummary.message_id == message_id)
    )

    retrieved_summaries = db.execute(statement).scalars().all()

    return retrieved_summaries


def get_chat_retrieved_summaries_by_knowledge_summary(
    db: Session,
    knowledge_summary_id: UUID
):
    statement = (
        select(ChatRetrievedSummary)
        .where(
            ChatRetrievedSummary.knowledge_summary_id
            == knowledge_summary_id
        )
    )

    retrieved_summaries = db.execute(statement).scalars().all()

    return retrieved_summaries


def delete_chat_retrieved_summary(
    db: Session,
    retrieved_summary_id: UUID
):
    retrieved_summary = get_chat_retrieved_summary(
        db,
        retrieved_summary_id
    )

    if retrieved_summary is None:
        return None

    db.delete(retrieved_summary)
    db.commit()

    return retrieved_summary