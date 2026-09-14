from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import Message, role_enum
from app.database.crud.chat import read_chat

# class Message(Base):
#     __tablename__ = "message"

#     id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"))
#     chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"))
#     order_in_chat: Mapped[int] = mapped_column(Integer, nullable=False)
#     role: Mapped[str] = mapped_column(SQLEnum(role_enum))
#     content: Mapped[str] = mapped_column(Text, nullable=False)
#     created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

#     user: Mapped["User"] = relationship(back_populates="message")
#     chat: Mapped["Chat"] = relationship(back_populates="message")
#     block_link: Mapped[List["MessageJoinBlock"]] = relationship(back_populates="message")
#     block: Mapped[List["MessageBlock"]] = relationship(secondary="message_join_block", viewonly=True)
#     retrieved_summary: Mapped[List["RetrievedSummary"]] = relationship(back_populates="message")
#     retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="message")
#     memory_detail: Mapped[List["MemoryDetail"]] = relationship(secondary="detail_join_message", viewonly=True)


def create_message(db: Session, user_id: UUID, chat_id: UUID, role: role_enum, content: str):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        return None

    order_in_chat = len(chat.message)+1

    message = Message(
        user_id = user_id,
        chat_id = chat_id,
        order_in_chat = order_in_chat,
        role = role,
        content = content
    )

    db.add(message)
    db.flush()
    db.refresh(message)

    chat.last_message_at = message.created_at

    db.commit()
    db.refresh(chat)

    return message


def read_message(db: Session, user_id: UUID, chat_id: UUID, message_id: UUID):
    chat = read_chat(db, user_id, chat_id)
    
    if chat is None:
        return None

    statement = select(Message).where(
        Message.id == message_id, Message.chat_id == chat_id, Message.user_id == user_id
    )

    message = db.execute(statement).scalar_one_or_none()

    return message


def delete_message(db: Session, user_id: UUID, chat_id: UUID, message_id: UUID):
    message = read_message(db, user_id, chat_id, message_id)

    if message is None:
        return None

    db.delete(message)
    db.commit()

    return message


def update_message(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID,
    content: str | None = None
):
    message = read_message(db, user_id, chat_id, message_id)

    if message is None:
        return None

    if content is not None:
        message.content = content

    db.commit()
    db.refresh(message)

    return message


def read_messages_after_order(db: Session, chat_id: UUID, after_order: int) -> list[Message]:
    """Read all messages in a chat with order_in_chat > after_order, sorted ascending."""
    statement = (
        select(Message)
        .where(Message.chat_id == chat_id, Message.order_in_chat > after_order)
        .order_by(Message.order_in_chat)
    )
    return list(db.execute(statement).scalars().all())