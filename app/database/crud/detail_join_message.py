from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import DetailJoinMessage

from app.database.crud.message import read_message
from app.database.crud.memory_detail import read_memory_detail

def create_detail_join_message(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    detail_id: UUID,
    message_id: UUID
):
    detail = read_memory_detail(
        db,
        user_id,
        detail_id
    )

    if detail is None:
        return None

    message = read_message(
        db,
        user_id,
        chat_id,
        message_id
    )

    if message is None:
        return None

    join = DetailJoinMessage(
        detail_id=detail_id,
        message_id=message_id
    )

    db.add(join)
    db.commit()
    db.refresh(join)

    return join

def read_detail_join_message(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    join_id: UUID
):
    statement = select(DetailJoinMessage).where(
        DetailJoinMessage.id == join_id
    )

    join = db.execute(statement).scalar_one_or_none()

    if join is None:
        return None

    message = read_message(
        db,
        user_id,
        chat_id,
        join.message_id
    )

    if message is None:
        return None

    return join

def read_detail_join_messages_by_detail(
    db: Session,
    user_id: UUID,
    detail_id: UUID
):
    detail = read_memory_detail(
        db,
        user_id,
        detail_id
    )

    if detail is None:
        return None

    statement = (
        select(DetailJoinMessage)
        .where(
            DetailJoinMessage.detail_id == detail_id
        )
    )

    joins = db.execute(statement).scalars().all()

    return joins

def read_detail_join_messages_by_message(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID
):
    message = read_message(
        db,
        user_id,
        chat_id,
        message_id
    )

    if message is None:
        return None

    statement = (
        select(DetailJoinMessage)
        .where(
            DetailJoinMessage.message_id == message_id
        )
    )

    joins = db.execute(statement).scalars().all()

    return joins

def delete_detail_join_message(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    join_id: UUID
):
    join = read_detail_join_message(
        db,
        user_id,
        chat_id,
        join_id
    )

    if join is None:
        return None

    db.delete(join)
    db.commit()

    return join

