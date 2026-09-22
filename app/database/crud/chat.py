from sqlalchemy.orm import Session
from sqlalchemy import select
from app.database.model import Chat
from app.database.crud.user import read_user
from datetime import datetime
from uuid import UUID

# class Chat(Base):
#     __tablename__ = "chat"

#     id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"))
#     title: Mapped[str] = mapped_column(Text, default="New Chat")
#     created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
#     last_message_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
#     last_summarisation_message_order: Mapped[int] = mapped_column(Integer, default = 0)
#     last_ingestion_message_order: Mapped[int] = mapped_column(Integer, default = 0)

#     user: Mapped["User"] = relationship(back_populates="chat")
#     message: Mapped[List["Message"]] = relationship(back_populates="chat")
#     message_block: Mapped[List["MessageBlock"]] = relationship(back_populates="chat")
#     memory_detail: Mapped[List["MemoryDetail"]] = relationship(back_populates="chat")
#     retrieved_summary: Mapped[List["RetrievedSummary"]] = relationship(back_populates="chat")
#     retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="chat")

def create_chat(db: Session, user_id : UUID, title: str | None = None):
    user = read_user(db, user_id)

    if user is None:
        return None

    if title is not None:
        title = title.strip()
        if not title:
            title = None

    if title is not None:
        chat = Chat(user_id = user_id, title = title)
    else:
        chat = Chat(user_id = user_id)

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return chat

def read_chat(db: Session, user_id: UUID, chat_id: UUID):
    user = read_user(db, user_id)
    
    if user is None:
        return None

    statement = select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    chat = db.execute(statement).scalar_one_or_none()

    return chat

def read_chats(db: Session, user_id: UUID) -> list[Chat]:
    user = read_user(db, user_id)

    if user is None:
        return []

    statement = (
        select(Chat)
        .where(Chat.user_id == user_id)
        .order_by(Chat.last_message_at.desc().nulls_last(), Chat.created_at.desc())
    )
    chats = db.execute(statement).scalars().all()

    return list(chats)


def delete_chat(db: Session, user_id: UUID, chat_id: UUID):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        return None

    # Use raw SQL so DB-level CASCADE handles children without ORM nullify issues
    from sqlalchemy import text as _text
    db.execute(_text("DELETE FROM chat WHERE id = :cid"), {"cid": str(chat_id)})
    db.commit()
    return chat

def update_chat(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    title: str | None = None,
    last_message_time: datetime | None = None,
    last_summarisation_message_order: int | None = None,
    last_ingestion_message_order: int | None = None,
    summarisation_going_on: bool | None = None,
    ingestion_going_on: bool | None = None,
):

    user = read_user(db, user_id)

    if user is None:
        return None

    chat = read_chat(db, user_id, chat_id)

    if chat is None:
      return None

    if last_message_time is not None:
        chat.last_message_at = last_message_time

    if title is not None:
        chat.title = title

    if last_ingestion_message_order is not None:
        chat.last_ingestion_message_order = last_ingestion_message_order

    if last_summarisation_message_order is not None:
        chat.last_summarisation_message_order = last_summarisation_message_order

    if summarisation_going_on is not None:
        chat.summarisation_going_on = summarisation_going_on

    if ingestion_going_on is not None:
        chat.ingestion_going_on = ingestion_going_on

    db.commit()
    db.refresh(chat)
    return chat
