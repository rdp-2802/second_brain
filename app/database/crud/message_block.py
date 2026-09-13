from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import MessageBlock
from app.database.crud.chat import read_chat


def create_message_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    content: str | None
):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        return None

    order_in_chat = len(chat.message_block) + 1

    block = MessageBlock(
        user_id = user_id,
        chat_id = chat_id,
        order_in_chat = order_in_chat,
        content = content
    )

    db.add(block)
    db.commit()
    db.refresh(block)

    return block


def read_message_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    block_id: UUID
):
    if read_chat(db, user_id, chat_id) is None:
        return None

    statement = select(MessageBlock).where(
        MessageBlock.id == block_id, MessageBlock.chat_id == chat_id
    )

    block = db.execute(statement).scalar_one_or_none()

    return block


def read_message_blocks_by_chat(
    db: Session,
    user_id: UUID,
    chat_id: UUID
):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        return None

    statement = (
        select(MessageBlock)
        .where(MessageBlock.chat_id == chat_id)
        .order_by(MessageBlock.order_in_chat)
    )

    blocks = db.execute(statement).scalars().all()

    return blocks


def delete_message_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    block_id: UUID
):
    block = read_message_block(
        db,
        user_id,
        chat_id,
        block_id
    )

    if block is None:
        return None

    db.delete(block)
    db.commit()

    return block


def update_message_block(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    block_id: UUID,
    content: str | None = None
):
    block = read_message_block(
        db,
        user_id,
        chat_id,
        block_id
    )

    if block is None:
        return None

    if content is not None:
        block.content = content

    db.commit()
    db.refresh(block)

    return block