from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import MessageJoinBlock
from app.database.crud.chat import read_chat
from app.database.crud.message import read_message


def create_message_join_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    block_id: UUID,
    message_id: UUID
):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        return None

    join = MessageJoinBlock(
        chat_id=chat_id,
        block_id=block_id,
        message_id=message_id
    )

    db.add(join)
    db.commit()
    db.refresh(join)

    return join


def read_message_join_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    join_id: UUID
):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        return None

    statement = select(MessageJoinBlock).where(
        MessageJoinBlock.id == join_id,
        MessageJoinBlock.chat_id == chat_id
    )

    join = db.execute(statement).scalar_one_or_none()

    return join


def read_message_join_blocks_by_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    block_id: UUID
):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        return None

    statement = (
        select(MessageJoinBlock)
        .where(
            MessageJoinBlock.block_id == block_id,
            MessageJoinBlock.chat_id == chat_id
        )
    )

    joins = db.execute(statement).scalars().all()

    return joins


def read_message_join_blocks_by_message(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID
):
    message = read_message(db, user_id, chat_id, message_id)

    if chat is None:
        return None

    statement = (
        select(MessageJoinBlock)
        .where(
            MessageJoinBlock.message_id == message_id,
        )
    )

    joins = db.execute(statement).scalars().all()

    return joins

def read_message_join_block_by_block_and_message(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    block_id: UUID,
    message_id: UUID
):
    block = get_message_block(db, user_id, chat_id, block_id)

    if block is None:
        return None

    message = read_message(db, user_id, chat_id, message_id)

    if message is None:
        return None

    statement = select(MessageJoinBlock).where(
        MessageJoinBlock.block_id == block_id,
        MessageJoinBlock.message_id == message_id
    )

    join = db.execute(statement).scalar_one_or_none()

    return join


def delete_message_join_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    join_id: UUID
):
    join = read_message_join_block(
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


def update_message_join_block(
    db: Session,
    join_id: UUID,
    chat_id: UUID | None = None,
    block_id: UUID | None = None,
    message_id: UUID | None = None
):
    join = get_message_join_block(
        db,
        join_id
    )

    if join is None:
        return None

    if chat_id is not None:
        join.chat_id = chat_id

    if block_id is not None:
        join.block_id = block_id

    if message_id is not None:
        join.message_id = message_id

    db.commit()
    db.refresh(join)

    return join