from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.model import RetrievedDetail
from app.database.crud.chat import read_chat
from app.database.crud.message import read_message
from app.database.crud.memory_summary import read_memory_summary
from app.database.crud.memory_detail import read_memory_detail


def create_retrieved_detail(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID,
    memory_detail_id: UUID
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


    memory_detail = read_memory_detail(
        db,
        user_id,
        memory_detail_id
    )

    if memory_detail is None:
        return None

    memory_summary_id = memory_detail.summary_id

    retrieved_detail = RetrievedDetail(
        user_id=user_id,
        chat_id=chat_id,
        message_id=message_id,
        memory_summary_id=memory_summary_id,
        memory_detail_id=memory_detail_id
    )

    db.add(retrieved_detail)
    db.commit()
    db.refresh(retrieved_detail)

    return retrieved_detail


def read_retrieved_detail(
    db: Session,
    user_id: UUID,
    retrieved_detail_id: UUID
):
    statement = select(RetrievedDetail).where(
        RetrievedDetail.id == retrieved_detail_id,
        RetrievedDetail.user_id == user_id
    )

    retrieved_detail = db.execute(
        statement
    ).scalar_one_or_none()

    return retrieved_detail


def read_retrieved_details_by_chat(
    db: Session,
    user_id: UUID,
    chat_id: UUID
):
    statement = (
        select(RetrievedDetail)
        .where(
            RetrievedDetail.user_id == user_id,
            RetrievedDetail.chat_id == chat_id
        )
    )

    retrieved_details = db.execute(
        statement
    ).scalars().all()

    return retrieved_details


def read_retrieved_details_by_message(
    db: Session,
    user_id: UUID,
    message_id: UUID
):
    statement = (
        select(RetrievedDetail)
        .where(
            RetrievedDetail.user_id == user_id,
            RetrievedDetail.message_id == message_id
        )
    )

    retrieved_details = db.execute(
        statement
    ).scalars().all()

    return retrieved_details


def read_retrieved_details_by_memory_summary(
    db: Session,
    user_id: UUID,
    memory_summary_id: UUID
):
    statement = (
        select(RetrievedDetail)
        .where(
            RetrievedDetail.user_id == user_id,
            RetrievedDetail.memory_summary_id == memory_summary_id
        )
    )

    retrieved_details = db.execute(
        statement
    ).scalars().all()

    return retrieved_details


def read_retrieved_details_by_memory_detail(
    db: Session,
    user_id: UUID,
    memory_detail_id: UUID
):
    statement = (
        select(RetrievedDetail)
        .where(
            RetrievedDetail.user_id == user_id,
            RetrievedDetail.memory_detail_id == memory_detail_id
        )
    )

    retrieved_details = db.execute(
        statement
    ).scalars().all()

    return retrieved_details


def delete_retrieved_detail(
    db: Session,
    user_id: UUID,
    retrieved_detail_id: UUID
):
    retrieved_detail = read_retrieved_detail(
        db,
        user_id,
        retrieved_detail_id
    )

    if retrieved_detail is None:
        return None

    db.delete(retrieved_detail)
    db.commit()

    return retrieved_detail