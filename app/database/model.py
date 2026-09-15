from datetime import datetime
from enum import Enum
import os
import uuid
from typing import List, Optional
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    create_engine,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    func,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

load_dotenv()
sql_url = os.getenv("SQL_URL")
engine = create_engine(sql_url)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=True)

def get_db(SessionLocal):
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Enum Python Classes
class role_enum(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# SQLAlchemy ORM Model for DB
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(30), nullable=False)
    email: Mapped[str] = mapped_column(String(30), nullable=False)
    mobile_no: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chat: Mapped[List["Chat"]] = relationship(back_populates="user")
    message: Mapped[List["Message"]] = relationship(back_populates="user")
    message_block: Mapped[List["MessageBlock"]] = relationship(back_populates="user")
    memory_detail: Mapped[List["MemoryDetail"]] = relationship(back_populates="user")
    memory_summary: Mapped[List["MemorySummary"]] = relationship(back_populates="user")
    retrieved_summary: Mapped[List["RetrievedSummary"]] = relationship(back_populates="user")
    retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="user")

class Chat(Base):
    __tablename__ = "chat"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(Text, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_summarisation_message_order: Mapped[int] = mapped_column(Integer, default = 0)
    last_ingestion_message_order: Mapped[int] = mapped_column(Integer, default = 0)
    summarisation_going_on: Mapped[bool] = mapped_column(Boolean, default = False, nullable = False)
    ingestion_going_on: Mapped[bool] = mapped_column(Boolean, default = False, nullable = False)

    user: Mapped["User"] = relationship(back_populates="chat")
    message: Mapped[List["Message"]] = relationship(back_populates="chat")
    message_block: Mapped[List["MessageBlock"]] = relationship(back_populates="chat")
    memory_detail: Mapped[List["MemoryDetail"]] = relationship(back_populates="chat")
    retrieved_summary: Mapped[List["RetrievedSummary"]] = relationship(back_populates="chat")
    retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="chat")


class Message(Base):
    __tablename__ = "message"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id",ondelete="CASCADE"), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id",ondelete="CASCADE"), nullable=False)
    order_in_chat: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[role_enum] = mapped_column(SQLEnum(role_enum), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="message")
    chat: Mapped["Chat"] = relationship(back_populates="message")
    block_link: Mapped[List["MessageJoinBlock"]] = relationship(back_populates="message")
    block: Mapped[List["MessageBlock"]] = relationship(secondary="message_join_block", viewonly=True)
    retrieved_summary: Mapped[List["RetrievedSummary"]] = relationship(back_populates="message")
    retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="message")
    memory_detail: Mapped[List["MemoryDetail"]] = relationship(secondary="detail_join_message", viewonly=True)

    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "order_in_chat",
            name="uq_message_chat_order"
        ),
    )


class MessageBlock(Base):
    __tablename__ = "message_block"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id",ondelete="CASCADE"), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id",ondelete="CASCADE"), nullable=False)
    order_in_chat: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="message_block")
    chat: Mapped["Chat"] = relationship(back_populates="message_block")
    message_link: Mapped[List["MessageJoinBlock"]] = relationship(back_populates="block")
    message: Mapped[List["Message"]] = relationship(secondary="message_join_block", viewonly=True)

    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "order_in_chat",
            name="uq_message_block_chat_order"
        ),
    )


class MessageJoinBlock(Base):
    __tablename__ = "message_join_block"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id",ondelete="CASCADE"),nullable=False)
    block_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message_block.id",ondelete="CASCADE"),nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id",ondelete="CASCADE"),nullable=False)

    block: Mapped["MessageBlock"] = relationship(back_populates="message_link")
    message: Mapped["Message"] = relationship(back_populates="block_link")

    __table_args__ = (
        UniqueConstraint("block_id", "message_id", name="uq_message_join_block_block_message"),
    )


class MemorySummary(Base):
    __tablename__ = "memory_summary"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id",ondelete="CASCADE"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(1024), nullable=False)
    retrieval_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    memory_detail: Mapped[List["MemoryDetail"]] = relationship(back_populates="memory_summary")
    retrieved_summary: Mapped[List["RetrievedSummary"]] = relationship(back_populates="memory_summary")
    retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="memory_summary")
    user: Mapped["User"] = relationship(back_populates="memory_summary")

    __table_args__ = (
        Index(
            "memory_summary_index",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )

class DetailJoinMessage(Base):
    __tablename__ = "detail_join_message"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detail_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("memory_detail.id",ondelete="CASCADE"), nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("message.id",ondelete="CASCADE"), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "detail_id",
            "message_id",
            name="uq_detail_join_message_detail_message"
            ),
    )

class MemoryDetail(Base):
    __tablename__ = "memory_detail"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id",ondelete="CASCADE"), nullable=False)
    summary_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("memory_summary.id",ondelete="CASCADE"), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id",ondelete="SET NULL"),nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    memory_summary: Mapped["MemorySummary"] = relationship(back_populates="memory_detail")
    chat: Mapped[Optional["Chat"]] = relationship(back_populates="memory_detail")
    user: Mapped["User"] = relationship(back_populates="memory_detail")
    retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="memory_detail")
    message: Mapped[List["Message"]] = relationship(secondary="detail_join_message", viewonly=True)

    __table_args__ = (
        Index(
            "memory_detail_index",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )


class RetrievedSummary(Base):
    __tablename__ = "retrieved_summary"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id",ondelete="CASCADE"), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id",ondelete="CASCADE"), nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id",ondelete="CASCADE"), nullable=False)
    memory_summary_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("memory_summary.id",ondelete="CASCADE"), nullable=False)

    user: Mapped["User"] = relationship(back_populates="retrieved_summary")
    chat: Mapped["Chat"] = relationship(back_populates="retrieved_summary")
    message: Mapped["Message"] = relationship(back_populates="retrieved_summary")
    memory_summary: Mapped["MemorySummary"] = relationship(back_populates="retrieved_summary")


class RetrievedDetail(Base):
    __tablename__ = "retrieved_detail"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id",ondelete="CASCADE"), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id",ondelete="CASCADE"), nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id",ondelete="CASCADE"), nullable=False)
    memory_summary_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("memory_summary.id",ondelete="CASCADE"), nullable=False)
    memory_detail_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("memory_detail.id",ondelete="CASCADE"), nullable=False)

    user: Mapped["User"] = relationship(back_populates="retrieved_detail")
    chat: Mapped["Chat"] = relationship(back_populates="retrieved_detail")
    message: Mapped["Message"] = relationship(back_populates="retrieved_detail")
    memory_summary: Mapped["MemorySummary"] = relationship(back_populates="retrieved_detail")
    memory_detail: Mapped["MemoryDetail"] = relationship(back_populates="retrieved_detail")