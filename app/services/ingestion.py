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
        indent=2,
    )

    memories_json = json.dumps(
        [memory.model_dump(mode="json") for memory in previous_memories],
        ensure_ascii=False,
        indent=2,
    )

    # Previous conversation context: blocks + gap messages (if any)
    summary_blocks_json = json.dumps(previous_summary_blocks, ensure_ascii=False)
    gap_messages_json = json.dumps(previous_gap_messages, ensure_ascii=False)

    prompt_1 = f"""
    # Memory Ingestion

    ## Brief

    You are a Memory Ingestion LLM for a personal knowledge base.

    Extract durable and useful NEW knowledge about the user from the Current
    Conversation ONLY. First create the Knowledge Details, then decide whether
    those Details belong to an existing Knowledge Summary or require a new Summary.

    A Summary represents a broad, persistent cluster of related knowledge.
    A Detail represents specific new knowledge and evidence from the current
    conversation.

    Do not summarize the conversation. Store only knowledge that can be useful
    for understanding the user in future conversations. Most conversations
    contain nothing worth storing — returning [] is the expected outcome far
    more often than not.

    IMPORTANT: Extract knowledge ONLY from the Current Conversation. Previous
    Conversation Summary and Gap Messages are context only — do not extract
    knowledge from them.

    ---

    ## Input

    ### User (user_name)
    {user_name}

    ### Current Date-Time (datetime)
    {current_datetime}

    Use this when interpreting relative time expressions such as "today",
    "yesterday", "last week", etc.

    ### Previous Conversation Summary

    {summary_blocks_json}

    These are compressed summaries of earlier parts of the conversation.
    Use them to understand the broader context and trajectory of the
    conversation. Do not treat them as current knowledge — they are
    background context only.

    ### Gap Messages

    {gap_messages_json}

    These are messages that were not included in any previous summary block.
    They provide additional context between the last summary and the current
    conversation. Do not extract knowledge from these — they are context only.

    ### Current Conversation

    {conversation_json}

    Each message contains:
    - id: message UUID
    - role: role_enum(User, Agent, System)
    - content: message content

    This is the ONLY section you should extract knowledge from.

    ### Previous Similar Memories

    {memories_json}

    Each memory contains:
    - summary_id: summary UUID
    - summary_content: summary text
    - details: list of detail texts

    Previous memories are context only. New Details must always be derived from
    the Current Conversation.

    ---
    """
    prompt_2 = """
    ## Task

    1. Read the Current Conversation and identify all meaningful, durable NEW
    knowledge about the user. Ask: would this still matter in three months?
    If not, it does not belong here.

    2. Create the Knowledge Details FIRST. Each Detail should explain the new
    information clearly and preserve useful context, reasoning, experience,
    change, or intention. Focus on what is NEW rather than repeating context
    already captured by Previous Similar Memories.

    3. For every Detail, provide `source_message_ids` — UUIDs of Current
    Conversation messages that directly support that Detail. Never use IDs
    from Previous Similar Memories and never invent IDs.

    4. After creating the Details, decide where each Detail belongs:
    - If it belongs to an existing Summary, use that Summary's `summary_id`
        and rewrite `summary_content` to incorporate the new development.
    - If it does not belong to any existing Summary, create a new Summary
        with `summary_id = null`.
    - If a Detail is genuinely relevant to more than one existing Summary,
        attach it to the single Summary it is MOST central to. Do not duplicate
        the same Detail across multiple Summaries.

    5. When new information contradicts or supersedes older information in a
    Summary, do not silently overwrite the old state. Rewrite the Summary to
    reflect the trajectory — what was true before, what changed, and when —
    so the arc stays legible. Compress older, less load-bearing detail as
    needed to make room; do not just append onto an ever-growing paragraph.
    A Summary should stay tight enough to remain a single coherent cluster —
    if it starts covering genuinely distinct topics, that is a signal it
    should split into separate Summaries rather than keep growing.

    6. A conversation can produce multiple Details and multiple Summaries when
    it contains genuinely different durable knowledge.

    7. Use dates when they provide useful temporal context. If the user says
    something like "yesterday" and the current date-time allows you to derive
    the approximate date, state the actual date. Do not invent temporal
    precision beyond what's derivable.

    8. Write `summary_content` and `detail_content` in the same language the
    user is writing in. Do not translate to English by default.

    9. If nothing meaningful should be remembered, return `[]`. This is common
    and correct — do not manufacture a Summary or Detail to avoid returning
    an empty list.

    ---

    ## Do's

    - Make Summaries broad enough to represent an ongoing knowledge cluster, but
    tight enough to stay coherent — split rather than let one Summary absorb
    unrelated threads.
    - Make Details sufficiently explanatory to preserve the important new
    information, but proportional to how significant the information actually
    is — a minor update deserves a short Detail, not an inflated one.
    - Update an existing Summary when the underlying knowledge is genuinely the
    same; preserve the trajectory when it has changed or been contradicted.
    - Create a new Summary when the underlying knowledge is genuinely different.
    - Use the user's own reasoning, wording, and language when it is important.
    - Treat user messages as the source of truth.

    ## Don'ts

    - Do not summarize the whole conversation.
    - Do not store casual conversation, temporary information, generic facts, or
    information with little future value.
    - Do not manufacture or over-infer information.
    - Do not treat assistant messages as evidence of the user's knowledge,
    thoughts, experiences, goals, or preferences.
    - Do not force unrelated knowledge into an existing Summary, and do not
    duplicate one Detail across multiple Summaries.
    - Do not unnecessarily repeat old context in a new Detail.
    - Do not use source message IDs from Previous Similar Memories.
    - Do not invent message IDs or dates.
    - Do not inflate the length or emotional register of a Detail or Summary
    beyond what the conversation actually supports — a technical preference
    and a significant life event should not read the same.

    ---

    ## Output

    Return ONLY a JSON list of OutputMemory objects — no explanations, reasoning,
    markdown, or text outside the list. Each OutputMemory has:

    - summary_id: the existing summary's UUID as a string, or null if this is a
    new Summary
    - summary_content: the full rewritten summary text (str)
    - details: a list of OutputDetail objects, each with:
        - detail_content: the detail text (str)
        - source_message_ids: a list of UUID strings from the Current
        Conversation that support this detail

    Example shape:

    [
        {{
            "summary_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "summary_content": "string",
            "details": [
                {{
                    "detail_content": "string",
                    "source_message_ids": ["11111111-1111-4111-8111-111111111111"]
                }}
            ]
        }}
    ]

    For a new Summary, set summary_id to null. Every Detail must contain at
    least one valid source message ID from the Current Conversation — never
    invent IDs, and never reuse IDs from Previous Similar Memories. If nothing
    is worth remembering, return an empty list: []
    """
    prompt = prompt_1 + prompt_2
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
    if chat.ingestion_going_on:
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

    # Parse + persist
    memories = _response_to_pydantic(response)
    _db_memory_ingestion(db, memories, user_id, chat_id)

    # Update last ingestion order
    last_order = max(m.order_in_chat for m in ingestion_messages)
    update_chat(db, user_id, chat_id, last_ingestion_message_order=last_order)
