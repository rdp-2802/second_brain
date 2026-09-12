from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import ChatRetrievedDetail


def add_chat_retrieved_detail(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID,
    knowledge_summary_id: UUID,
    knowledge_detail_id: UUID,
    retrieved_summary_id: UUID | None = None
):
    retrieved_detail = ChatRetrievedDetail(
        user_id=user_id,
        chat_id=chat_id,
        message_id=message_id,
        knowledge_summary_id=knowledge_summary_id,
        knowledge_detail_id=knowledge_detail_id,
        retrieved_summary_id=retrieved_summary_id
    )

    db.add(retrieved_detail)
    db.commit()
    db.refresh(retrieved_detail)

    return retrieved_detail


def get_chat_retrieved_detail(
    db: Session,
    retrieved_detail_id: UUID
):
    statement = select(ChatRetrievedDetail).where(
        ChatRetrievedDetail.id == retrieved_detail_id
    )

    retrieved_detail = db.execute(statement).scalar_one_or_none()

    return retrieved_detail


def get_chat_retrieved_details_by_chat(
    db: Session,
    chat_id: UUID
):
    statement = (
        select(ChatRetrievedDetail)
        .where(ChatRetrievedDetail.chat_id == chat_id)
    )

    retrieved_details = db.execute(statement).scalars().all()

    return retrieved_details


def get_chat_retrieved_details_by_message(
    db: Session,
    message_id: UUID
):
    statement = (
        select(ChatRetrievedDetail)
        .where(ChatRetrievedDetail.message_id == message_id)
    )

    retrieved_details = db.execute(statement).scalars().all()

    return retrieved_details


def get_chat_retrieved_details_by_knowledge_summary(
    db: Session,
    knowledge_summary_id: UUID
):
    statement = (
        select(ChatRetrievedDetail)
        .where(
            ChatRetrievedDetail.knowledge_summary_id
            == knowledge_summary_id
        )
    )

    retrieved_details = db.execute(statement).scalars().all()

    return retrieved_details


def get_chat_retrieved_details_by_knowledge_detail(
    db: Session,
    knowledge_detail_id: UUID
):
    statement = (
        select(ChatRetrievedDetail)
        .where(
            ChatRetrievedDetail.knowledge_detail_id
            == knowledge_detail_id
        )
    )

    retrieved_details = db.execute(statement).scalars().all()

    return retrieved_details


def get_chat_retrieved_details_by_summary(
    db: Session,
    retrieved_summary_id: UUID
):
    statement = (
        select(ChatRetrievedDetail)
        .where(
            ChatRetrievedDetail.retrieved_summary_id
            == retrieved_summary_id
        )
    )

    retrieved_details = db.execute(statement).scalars().all()

    return retrieved_details


def delete_chat_retrieved_detail(
    db: Session,
    retrieved_detail_id: UUID
):
    retrieved_detail = get_chat_retrieved_detail(
        db,
        retrieved_detail_id
    )

    if retrieved_detail is None:
        return None

    db.delete(retrieved_detail)
    db.commit()

    return retrieved_detail