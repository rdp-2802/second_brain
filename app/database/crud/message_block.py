from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.model import MessageBlock
from app.database.crud.chat import read_chat


# class MessageBlock(Base):
#     __tablename__ = "message_block"

#     id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     user_id: ----
#     chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"))
#     order_in_chat: Mapped[int] = mapped_column(Integer)
#     summary_text: Mapped[str] = mapped_column(Text)
#     created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
#     user: ---
#     chat: Mapped["Chat"] = relationship(back_populates="message_block")
#     message_link: Mapped[List["MessageJoinBlock"]] = relationship(back_populates="block")
#     message: Mapped[List["Message"]] = relationship(secondary="message_join_block", viewonly=True)


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


def get_message_block(
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
    block = get_message_block(
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
    block = get_message_block(
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