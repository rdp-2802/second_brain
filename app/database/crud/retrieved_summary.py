from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import RetrievedSummary
from app.database.crud.message import read_message
from app.database.crud.chat import read_chat
from app.database.crud.memory_summary import read_memory_summary

def create_retrieved_summary(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID,
    memory_summary_id: UUID
):
    chat = read_chat(
        db,
        user_id,
        chat_id
    )

    if chat is None:
        return None

    message = read_message(
        db,
        user_id,
        chat_id,
        message_id
    )

    if message is None:
        return None

    memory_summary = read_memory_summary(
        db,
        user_id,
        memory_summary_id
    )

    if memory_summary is None:
        return None

    retrieved_summary = RetrievedSummary(
        user_id=user_id,
        chat_id=chat_id,
        message_id=message_id,
        memory_summary_id=memory_summary_id
    )

    db.add(retrieved_summary)
    db.commit()
    db.refresh(retrieved_summary)

    return retrieved_summary

def read_retrieved_summary(
    db: Session,
    user_id: UUID,
    retrieved_summary_id: UUID
):
    statement = select(RetrievedSummary).where(
        RetrievedSummary.id == retrieved_summary_id,
        RetrievedSummary.user_id == user_id
    )

    retrieved_summary = db.execute(
        statement
    ).scalar_one_or_none()

    return retrieved_summary


def read_retrieved_summaries_by_chat(
    db: Session,
    user_id: UUID,
    chat_id: UUID
):
    statement = (
        select(RetrievedSummary)
        .where(
            RetrievedSummary.user_id == user_id,
            RetrievedSummary.chat_id == chat_id
        )
    )

    retrieved_summaries = db.execute(
        statement
    ).scalars().all()

    return retrieved_summaries


def read_retrieved_summaries_by_message(
    db: Session,
    user_id: UUID,
    message_id: UUID
):
    statement = (
        select(RetrievedSummary)
        .where(
            RetrievedSummary.user_id == user_id,
            RetrievedSummary.message_id == message_id
        )
    )

    retrieved_summaries = db.execute(
        statement
    ).scalars().all()

    return retrieved_summaries


def read_retrieved_summaries_by_memory_summary(
    db: Session,
    user_id: UUID,
    memory_summary_id: UUID
):
    statement = (
        select(RetrievedSummary)
        .where(
            RetrievedSummary.user_id == user_id,
            RetrievedSummary.memory_summary_id == memory_summary_id
        )
    )

    retrieved_summaries = db.execute(
        statement
    ).scalars().all()

    return retrieved_summaries


def delete_retrieved_summary(
    db: Session,
    user_id: UUID,
    retrieved_summary_id: UUID
):
    retrieved_summary = read_retrieved_summary(
        db,
        user_id,
        retrieved_summary_id
    )

    if retrieved_summary is None:
        return None

    db.delete(retrieved_summary)
    db.commit()

    return retrieved_summary