from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, DateTime, ForeignKey, func, Float, Integer, Boolean, UniqueConstraint
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from enum import Enum
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker
import uuid
import os
from dotenv import load_dotenv


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


# class knowledge_type_enum(Enum):
#     STORY = "story"
#     HABIT = "habit"
#     BELIEF = "belief"
#     GOAL = "goal"
#     PREFERENCE = "preference"


# --------------------------------

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "user"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(30), nullable=False)
    email: Mapped[str] = mapped_column(String(30), nullable=False)
    mobile_no: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chats: Mapped[List["Chats"]] = relationship(back_populates="user")
    user_knowledge_details: Mapped[List["KnowledgeDetail"]] = relationship(back_populates="knowledge_detail_user")
    user_knowledge_summary: Mapped[List["KnowledgeSummary"]] = relationship(back_populates="knowledge_summary_user")
    retrieved_summaries: Mapped[List["ChatRetrievedSummary"]] = relationship(back_populates="user")
    retrieved_details: Mapped[List["ChatRetrievedDetail"]] = relationship(back_populates="user")


class Chats(Base):
    __tablename__ = "chat"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"))

    title: Mapped[str] = mapped_column(Text, default="New Chat")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_message_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_summary_message_order: Mapped[int] = mapped_column(Integer, default = 0)
    last_memory_extracted_message_order: Mapped[int] = mapped_column(Integer, default = 0)

    user: Mapped["User"] = relationship(back_populates="chats")

    messages: Mapped[List["Messages"]] = relationship(back_populates="chat")
    message_blocks: Mapped[List["MessageBlock"]] = relationship(back_populates="chat")
    knowledge_details: Mapped[List["KnowledgeDetail"]] = relationship(back_populates="chat")
    retrieved_summaries: Mapped[List["ChatRetrievedSummary"]] = relationship(back_populates="chat")
    retrieved_details: Mapped[List["ChatRetrievedDetail"]] = relationship(back_populates="chat")


class Messages(Base):
    __tablename__ = "message"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"))

    order_in_chat: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(SQLEnum(role_enum))
    content: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    context_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")

    chat: Mapped["Chats"] = relationship(back_populates="messages")

    # Association-object rows linking this message into whichever block(s) it belongs to.
    block_links: Mapped[List["MessageJoinBlock"]] = relationship(back_populates="message")
    # Convenience read-only shortcut straight to the MessageBlock rows themselves.
    blocks: Mapped[List["MessageBlock"]] = relationship(
        secondary="message_join_block", viewonly=True
    )

    retrieved_summaries: Mapped[List["ChatRetrievedSummary"]] = relationship(back_populates="message")
    retrieved_details: Mapped[List["ChatRetrievedDetail"]] = relationship(back_populates="message")


class MessageBlock(Base):
    __tablename__ = "message_block"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"))

    order_in_chat: Mapped[int] = mapped_column(Integer)

    summary_text: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chat: Mapped["Chats"] = relationship(back_populates="message_blocks")

    # Association-object rows for each message in this block.
    message_links: Mapped[List["MessageJoinBlock"]] = relationship(back_populates="block")
    # Convenience read-only shortcut straight to the Messages rows themselves.
    messages: Mapped[List["Messages"]] = relationship(
        secondary="message_join_block", viewonly=True
    )


class MessageJoinBlock(Base):
    __tablename__ = "message_join_block"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"))
    block_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message_block.id"))
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id"))

    block: Mapped["MessageBlock"] = relationship(back_populates="message_links")
    message: Mapped["Messages"] = relationship(back_populates="block_links")

    __table_args__ = (
        # Prevents the same message from being linked into the same block twice.
        UniqueConstraint("block_id", "message_id", name="uq_message_join_block_block_message"),
    )


class KnowledgeSummary(Base):
    __tablename__ = "knowledge_summary"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    # knowledge_type: Mapped[str] = mapped_column(SQLEnum(knowledge_type_enum))
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(1024))

    relevancy_score: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    details: Mapped[List["KnowledgeDetail"]] = relationship(back_populates="summary")
    retrieved_summaries: Mapped[List["ChatRetrievedSummary"]] = relationship(back_populates="knowledge_summary")
    retrieved_details: Mapped[List["ChatRetrievedDetail"]] = relationship(back_populates="knowledge_summary")
    knowledge_summary_user: Mapped["User"] = relationship(back_populates="user_knowledge_summary")

    __table_args__ = (
        Index(
            "knowledge_summary_index",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )


class KnowledgeDetail(Base):
    __tablename__ = "knowledge_detail"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"))
    summary_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_summary.id"))
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"))

    source_message_ids: Mapped[List[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))

    detail_content: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[List[float]] = mapped_column(Vector(1024))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    summary: Mapped["KnowledgeSummary"] = relationship(back_populates="details")
    chat: Mapped["Chats"] = relationship(back_populates="knowledge_details")
    knowledge_detail_user: Mapped["User"] = relationship(back_populates="user_knowledge_details")
    retrieved_details: Mapped[List["ChatRetrievedDetail"]] = relationship(back_populates="knowledge_detail")

    __table_args__ = (
        Index(
            "knowledge_detail_index",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )


class ChatRetrievedSummary(Base):
    __tablename__ = "chat_retrieved_summary"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"), nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id"), nullable=False)
    knowledge_summary_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_summary.id"), nullable=False)

    user: Mapped["User"] = relationship(back_populates="retrieved_summaries")
    chat: Mapped["Chats"] = relationship(back_populates="retrieved_summaries")
    message: Mapped["Messages"] = relationship(back_populates="retrieved_summaries")
    knowledge_summary: Mapped["KnowledgeSummary"] = relationship(back_populates="retrieved_summaries")
    retrieved_details: Mapped[List["ChatRetrievedDetail"]] = relationship(
        back_populates="retrieved_summary",
        foreign_keys="ChatRetrievedDetail.retrieved_summary_id"
    )


class ChatRetrievedDetail(Base):
    __tablename__ = "chat_retrieved_detail"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id"), nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id"), nullable=False)
    knowledge_summary_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_summary.id"), nullable=False)
    knowledge_detail_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_detail.id"), nullable=False)
    retrieved_summary_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_retrieved_summary.id"))

    user: Mapped["User"] = relationship(back_populates="retrieved_details")
    chat: Mapped["Chats"] = relationship(back_populates="retrieved_details")
    message: Mapped["Messages"] = relationship(back_populates="retrieved_details")
    knowledge_summary: Mapped["KnowledgeSummary"] = relationship(back_populates="retrieved_details")
    knowledge_detail: Mapped["KnowledgeDetail"] = relationship(back_populates="retrieved_details")
    retrieved_summary: Mapped[Optional["ChatRetrievedSummary"]] = relationship(
        back_populates="retrieved_details",
        foreign_keys=[retrieved_summary_id]
    )