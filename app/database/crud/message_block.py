from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import MessageBlock
from app.database.crud.chat import get_chat


def add_message_block(
    db: Session,
    chat_id: UUID,
    summary_text: str | None
):
    chat = get_chat(db, chat_id)

    if chat is None:
        return None

    chat_msg_blocks = len(chat.message_blocks)
    order_in_chat = chat_msg_blocks + 1

    block = MessageBlock(
        chat_id=chat_id,
        order_in_chat=order_in_chat,
        summary_text=summary_text
    )

    db.add(block)
    db.commit()
    db.refresh(block)

    return block


def get_message_block(
    db: Session,
    block_id: UUID
):
    statement = select(MessageBlock).where(
        MessageBlock.id == block_id
    )

    block = db.execute(statement).scalar_one_or_none()

    return block


def get_message_blocks_by_chat(
    db: Session,
    chat_id: UUID
):
    statement = (
        select(MessageBlock)
        .where(MessageBlock.chat_id == chat_id)
        .order_by(MessageBlock.order_in_chat)
    )

    blocks = db.execute(statement).scalars().all()

    return blocks


def delete_message_block(
    db: Session,
    block_id: UUID
):
    block = get_message_block(
        db,
        block_id
    )

    if block is None:
        return None

    db.delete(block)
    db.commit()

    return block


def update_message_block(
    db: Session,
    block_id: UUID,
    summary_text: str | None = None
):
    block = get_message_block(
        db,
        block_id
    )

    if block is None:
        return None

    if summary_text is not None:
        block.summary_text = summary_text

    db.commit()
    db.refresh(block)

    return block