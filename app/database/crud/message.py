from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import Messages, role_enum
from app.database.crud.chat import get_chat


def add_message(db: Session, user_id: UUID, chat_id: UUID, role: role_enum, content: str):
    chat = get_chat(db, user_id, chat_id)

    if chat is None:
        return None

    order_in_chat = len(chat.messages)+1

    message = Messages(
        chat_id=chat_id,
        order_in_chat=order_in_chat,
        role=role,
        content=content
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    chat.last_message_at = message.created_at

    db.commit()
    db.refresh(chat)

    return message


def get_message(db: Session, message_id: UUID):
    statement = select(Messages).where(
        Messages.id == message_id
    )

    message = db.execute(statement).scalar_one_or_none()

    return message

def update_context_required(db: Session, message_id: UUID, context_required: bool):
    statement = select(Messages).where(Messages.id == message_id)
    message = db.execute(statement).scalar_one_or_none()
    if message is None:
        return None
    message.context_required = context_required
    db.commit()
    db.refresh(message)
    return message


def get_messages_by_chat(db: Session, user_id: UUID, chat_id: UUID):
    chat = get_chat(db, user_id, chat_id)

    if chat is None:
        return None

    statement = (
        select(Messages)
        .where(Messages.chat_id == chat_id)
        .order_by(Messages.order_in_chat)
    )

    messages = db.execute(statement).scalars().all()

    return messages


def delete_message(db: Session, message_id: UUID):
    message = get_message(db, message_id)

    if message is None:
        return None

    db.delete(message)
    db.commit()

    return message


def update_message(
    db: Session,
    message_id: UUID,
    content: str | None = None
):
    message = get_message(db, message_id)

    if message is None:
        return None

    if content is not None:
        message.content = content

    db.commit()
    db.refresh(message)

    return message