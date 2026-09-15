import json
from datetime import datetime
from pydantic import BaseModel, TypeAdapter
from uuid import UUID

from app.database.model import Message, MessageBlock, role_enum
from app.database.crud.user import read_user
from app.database.crud.memory_summary import create_memory_summary, update_memory_summary
from app.database.crud.memory_detail import create_memory_detail
from app.database.crud.detail_join_message import create_detail_join_message
from app.database.crud.chat import read_chat, update_chat
from app.database.crud.message import read_messages_after_order
from app.database.crud.message_block import read_message_blocks_by_chat
from sqlalchemy.orm import Session
from app.models.embedding import generate_embedding
from app.services.semantic_retrieval import retrieve_memory

try:
    from app.models.llm import chat_gemini
except ImportError:
    from app.services.llm import chat_gemini


class IngestionMessage(BaseModel):
    id: UUID
    role: role_enum
    content: str

    model_config = {"arbitrary_types_allowed": True}


class OutputDetail(BaseModel):
    detail_content: str
    source_message_ids: list[UUID]


class OutputMemory(BaseModel):
    summary_id: UUID | None
    summary_content: str
    details: list[OutputDetail]


def _messages_to_ingestion_format(messages: list[Message]) -> list[IngestionMessage]:
    """Convert raw Message objects to IngestionMessage for the prompt."""
    return [
        IngestionMessage(id=m.id, role=m.role, content=m.content)
        for m in messages
    ]


def _ingestion_message_embedding(ingestion_messages: list[IngestionMessage]):
    message_string = "\n".join(message.content for message in ingestion_messages)
    embedding = generate_embedding(message_string)
    return embedding


def _ingestion_prompt(
    user_name: str,
    ingestion_messages: list[IngestionMessage],
    previous_memories: list,
    previous_summary_blocks: list[dict],
    previous_gap_messages: list[dict],
) -> str:
    current_datetime = datetime.now()

    conversation_json = json.dumps(
        [message.model_dump(mode="json") for message in ingestion_messages],
        ensure_ascii=False,
    )
    memories_json = json.dumps(
        [memory.model_dump(mode="json") for memory in previous_memories],
        ensure_ascii=False,
    )
    summary_blocks_json = json.dumps(previous_summary_blocks, ensure_ascii=False)
    gap_messages_json = json.dumps(previous_gap_messages, ensure_ascii=False)

    prompt = f"""You are extracting memory from a chunk of a conversation
between an assistant and {user_name}. Current time: {current_datetime}.

HOW MEMORY IS STRUCTURED:
- memory_detail: a concrete, dated piece of what happened in the user's
  life at a specific point in time (an event, decision, plan, fact,
  feeling tied to a specific situation).
- memory_summary: a short label for a cluster of memory_details that are
  about the same recurring theme in the user's life (e.g. "relationship
  with X", "job search", "conflict with landlord"). Similar events can
  recur at different times — each occurrence is its own memory_detail,
  clustered under one memory_summary.

WHAT YOU'RE GIVEN:

1. Chat Context (Block Summaries + Gap Messages) — earlier parts of THIS
   conversation, given only so you understand what the Ingestion Messages
   below mean. Never extract memory_details from this.
   Block Summaries: {summary_blocks_json}
   Gap Messages: {gap_messages_json}

2. Ingestion Messages — the ONLY source you extract new memory_details
   from:
   {conversation_json}

3. Previous Memory Summaries (with their existing details) — given only
   so you can decide whether a new memory_detail belongs to one of these
   existing memory_summaries, or needs a new one. Never copy content from
   here into your output; it's for placement only.
   {memories_json}

WHAT TO DO:
1. Go through the Ingestion Messages and pull out concrete memory_details
   — things that happened, were decided, or were revealed about the user's
   life. Skip venting, small talk, and anything with no future relevance.
2. For each memory_detail, decide which memory_summary it belongs to:
   - If it's the same recurring theme as an existing memory_summary, use
     that summary_id and rewrite summary_content to reflect the fuller
     picture (including this new development).
   - If it's a new theme, set summary_id to null and write a fresh
     summary_content.
3. Every memory_detail needs source_message_ids — ids from Ingestion
   Messages only, never invented, never from Previous Memory Summaries.
4. If nothing in the Ingestion Messages is worth remembering, return [].

Return ONLY this JSON, nothing else:
[{{"summary_id": "<uuid or null>", "summary_content": "<string>", "details": [{{"detail_content": "<string>", "source_message_ids": ["<uuid>"]}}]}}]
"""
    return prompt


def _response_to_pydantic(response: str) -> list[OutputMemory]:
    adapter = TypeAdapter(list[OutputMemory])
    memories = adapter.validate_json(response)
    return memories


def _db_memory_ingestion(db: Session, memories: list[OutputMemory], user_id: UUID, chat_id: UUID):
    for memory in memories:
        if memory.summary_id is None:
            updated_memory = create_memory_summary(db, user_id, memory.summary_content)
        else:
            updated_memory = update_memory_summary(db, user_id, memory.summary_id, memory.summary_content)

        if updated_memory is None:
            continue

        for detail in memory.details:
            mem_detail = create_memory_detail(
                db, user_id, updated_memory.id, chat_id, detail.detail_content
            )
            if mem_detail is None:
                continue

            for message_id in detail.source_message_ids:
                create_detail_join_message(db, user_id, chat_id, mem_detail.id, message_id)


def _get_block_max_order(block: MessageBlock) -> int:
    """Get the max order_in_chat of messages linked to this block."""
    messages = getattr(block, "message", []) or []
    if not messages:
        return block.order_in_chat
    return max(m.order_in_chat for m in messages)


def _build_ingestion_context(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    last_ingested: int,
    ingestion_messages: list[Message],
) -> tuple[list[dict], list[dict]]:
    """
    Build previous conversation context for the ingestion prompt.
    Uses 3-case logic based on last_summarisation_message_order vs first ingestion order.

    Returns (previous_summary_blocks, gap_messages) as list[dict].
    """
    blocks = read_message_blocks_by_chat(db, user_id, chat_id) or []

    if not ingestion_messages:
        return [], []

    first_ingestion_order = ingestion_messages[0].order_in_chat

    # Format all blocks as previous context
    previous_summary_blocks = [
        {"order": block.order_in_chat, "summary": block.content}
        for block in blocks
    ]

    if not blocks:
        # No blocks yet — no gap, no previous summary
        return previous_summary_blocks, []

    # Get max message order from the last block
    last_block_max_order = _get_block_max_order(blocks[-1])

    # Read last_summarisation_message_order from chat
    chat = read_chat(db, user_id, chat_id)
    last_summarised = chat.last_summarisation_message_order or 0

    # Three cases
    if last_summarised + 1 == first_ingestion_order:
        # Case a: Clean — ingestion messages start right after last summarised
        # No gap messages needed
        return previous_summary_blocks, []

    elif last_summarised >= first_ingestion_order:
        # Case b: Overlap — summarisation went beyond ingestion start
        # The overlapping block provides context, no gap needed
        return previous_summary_blocks, []

    else:
        # Case c: Gap — messages between last block and ingestion window
        # Include gap messages as additional context
        gap_messages_raw = read_messages_after_order(db, chat_id, last_block_max_order)
        ingestion_ids = {m.id for m in ingestion_messages}
        gap_messages_raw = [m for m in gap_messages_raw if m.id not in ingestion_ids]

        gap_messages = [
            {
                "order": m.order_in_chat,
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
            }
            for m in gap_messages_raw
        ]

        return previous_summary_blocks, gap_messages


def ingestion(db: Session, user_id: UUID, chat_id: UUID) -> None:
    """
    Extract durable memories from messages.
    Loads its own context from DB. Returns None (stateless).

    1. Sanity check: chat.ingestion_going_on
    2. Load messages after last_ingestion_message_order
    3. Build prompt with 3-case block logic
    4. LLM call
    5. Persist memories
    6. Update last_ingestion_message_order
    """
    # Sanity check
    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        return

    last_ingested = chat.last_ingestion_message_order or 0

    # Load messages to ingest
    ingestion_messages = read_messages_after_order(db, chat_id, last_ingested)
    if not ingestion_messages:
        return

    # Build context with 3-case block logic
    previous_summary_blocks, gap_messages = _build_ingestion_context(
        db, user_id, chat_id, last_ingested, ingestion_messages
    )

    # Convert to ingestion format
    ingestion_format = _messages_to_ingestion_format(ingestion_messages)

    # Embedding for similarity search
    embedding = _ingestion_message_embedding(ingestion_format)

    # Retrieve previous similar memories
    summary_limit = 5
    detail_limit = 5
    previous_memories = retrieve_memory(db, user_id, embedding, summary_limit, detail_limit)

    # Get user name
    user = read_user(db, user_id)
    user_name = user.name if user else "User"

    # Build prompt
    prompt = _ingestion_prompt(
        user_name, ingestion_format, previous_memories,
        previous_summary_blocks, gap_messages,
    )
    # LLM call
    response = chat_gemini(prompt)

    print(f"Memory Ingested: {response}")
    print("# ---------------------------------------------------------------------------")

    # Parse + persist
    memories = _response_to_pydantic(response)
    _db_memory_ingestion(db, memories, user_id, chat_id)

    # Update last ingestion order
    last_order = max(m.order_in_chat for m in ingestion_messages)
    update_chat(db, user_id, chat_id, last_ingestion_message_order=last_order)
