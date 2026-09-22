import json
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.model import Message, MessageBlock, role_enum
from app.database.crud.chat import read_chat, update_chat
from app.database.crud.message import create_message
from app.database.crud.retrieved_summary import create_retrieved_summary
from app.database.crud.retrieved_detail import create_retrieved_detail
from app.database.crud.memory_summary import update_memory_summary_retrieval_metadata
from app.services.semantic_retrieval import semantic_retrieval, RetrievedMemory
from google.genai.errors import ClientError

try:
    from app.models.llm import chat_gemini
except ImportError:
    from app.services.llm import chat_gemini  # fallback for legacy path


# Pydantic models (PascalCase, allow ORM objects)
class SummaryBlock(BaseModel):
    order: int
    block: MessageBlock

    model_config = {"arbitrary_types_allowed": True}


class RecentMessage(BaseModel):
    order: int
    message: Message

    model_config = {"arbitrary_types_allowed": True}


class ChatUser(BaseModel):
    user_id: UUID
    chat_id: UUID


def _form_recent_messages(messages: list[Message]) -> list[RecentMessage]:
    items: list[RecentMessage] = [
        RecentMessage(order=m.order_in_chat, message=m)
        for m in messages
    ]
    items.sort(key=lambda x: x.order)
    return items


def _form_summary_blocks(blocks: list[MessageBlock]) -> list[SummaryBlock]:
    items: list[SummaryBlock] = []

    for block in blocks:
        sb = SummaryBlock(
            order=block.order_in_chat,
            block=block,
        )
        items.append(sb)

    items.sort(key=lambda x: x.order)
    return items


def _form_chat_context(
    db: Session, chat_user: ChatUser
) -> tuple[list[SummaryBlock], list[RecentMessage], list[Message]]:
    chat_id = chat_user.chat_id
    user_id = chat_user.user_id

    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        return [], [], []

    # New Chat: no messages and no blocks yet
    if not chat.message and not chat.message_block:
        return [], [], []

    # Summary blocks are ALL blocks (they represent already-summarised history)
    blocks: list[MessageBlock] = sorted(list(chat.message_block or []), key=lambda b: b.order_in_chat)
    summary_blocks = _form_summary_blocks(blocks)

    # Recent messages are only those AFTER last_summarisation_message_order
    last_summarised_order = chat.last_summarisation_message_order or 0
    raw_messages: list[Message] = [
        m for m in (chat.message or [])
        if m.order_in_chat > last_summarised_order
    ]
    raw_messages.sort(key=lambda m: m.order_in_chat)
    recent_messages = _form_recent_messages(raw_messages)

    return summary_blocks, recent_messages, raw_messages


# ---------------------------------------------------------------------------
# Memory extraction + deduplication (lives here — uses raw Message ORM objects)
# ---------------------------------------------------------------------------

def _extract_recent_memories(messages: list[Message]) -> list[RetrievedMemory]:
    """
    Extract all memories attached to recent messages (historical retrievals).
    Deduplicates by (memory_type, memory.id).
    """
    seen: set[tuple[str, UUID]] = set()
    memories: list[RetrievedMemory] = []

    for message in messages:
        for rs in getattr(message, "retrieved_summary", []) or []:
            mem = getattr(rs, "memory_summary", None)
            if mem is not None:
                key = ("summary", mem.id)
                if key not in seen:
                    seen.add(key)
                    memories.append(RetrievedMemory(
                        memory_type="summary",
                        memory=mem,
                        similarity=0.0,
                    ))

        for rd in getattr(message, "retrieved_detail", []) or []:
            mem = getattr(rd, "memory_detail", None)
            if mem is not None:
                key = ("detail", mem.id)
                if key not in seen:
                    seen.add(key)
                    memories.append(RetrievedMemory(
                        memory_type="detail",
                        memory=mem,
                        similarity=0.0,
                    ))

    return memories


def _deduplicate_previous_memories(
    previous_memories: list[RetrievedMemory],
    current_retrieved: list[RetrievedMemory],
) -> list[RetrievedMemory]:
    """
    Remove from previous_memories any memory that appears in current_retrieved.
    Current query always gets ALL its retrieved memories.
    """
    current_ids: set[tuple[str, UUID]] = {
        (m.memory_type, m.memory.id) for m in current_retrieved
    }
    return [
        m for m in previous_memories
        if (m.memory_type, m.memory.id) not in current_ids
    ]


def _persist_retrieval_audit(
    db: Session,
    user_id: UUID,
    chat_id: UUID,
    message_id: UUID,
    retrieved_memories: list[RetrievedMemory],
) -> None:
    """Write RetrievedSummary and RetrievedDetail rows for the current query."""
    for mem in retrieved_memories:
        if mem.memory_type == "summary":
            create_retrieved_summary(
                db,
                user_id=user_id,
                chat_id=chat_id,
                message_id=message_id,
                memory_summary_id=mem.memory.id,
            )
            update_memory_summary_retrieval_metadata(db, user_id, mem.memory.id)

        elif mem.memory_type == "detail":
            create_retrieved_detail(
                db,
                user_id=user_id,
                chat_id=chat_id,
                message_id=message_id,
                memory_detail_id=mem.memory.id,
            )


# ---------------------------------------------------------------------------
# LLM JSON + prompt
# ---------------------------------------------------------------------------

def _llm_json_creation(
    summary_blocks: list[SummaryBlock],
    recent_messages: list[RecentMessage],
    recent_memories: list[RetrievedMemory],
    query_retrieved_memories: list[RetrievedMemory],
) -> tuple[str, str, str, str]:
    llm_summary = [{"Summary Text": block.block.content} for block in summary_blocks]

    llm_recent_messages = [
        {
            "Role": item.message.role.value if hasattr(item.message.role, "value") else str(item.message.role),
            "Content": item.message.content,
        }
        for item in recent_messages
    ]

    llm_recent_memories: list[dict] = []
    for mem in recent_memories:
        if mem.memory_type == "summary":
            llm_recent_memories.append({
                "type": "summary",
                "content": mem.memory.content,
            })
        elif mem.memory_type == "detail":
            llm_recent_memories.append({
                "type": "detail",
                "content": mem.memory.content,
            })

    llm_query_memories: list[dict] = []
    for mem in query_retrieved_memories:
        if mem.memory_type == "summary":
            llm_query_memories.append({
                "similarity": round(mem.similarity, 4),
                "type": "summary",
                "content": mem.memory.content,
            })
        elif mem.memory_type == "detail":
            llm_query_memories.append({
                "similarity": round(mem.similarity, 4),
                "type": "detail",
                "content": mem.memory.content,
            })

    return (
        json.dumps(llm_summary, ensure_ascii=False),
        json.dumps(llm_recent_messages, ensure_ascii=False),
        json.dumps(llm_recent_memories, ensure_ascii=False),
        json.dumps(llm_query_memories, ensure_ascii=False),
    )


def _llm_prompt_creation(
    json_llm_summary: str,
    json_llm_recent_messages: str,
    json_llm_recent_memories: str,
    json_llm_query_memories: str,
    query: str
):
    system_prompt = """
    You are a personal conversational assistant.

    Your task is to respond naturally and helpfully to the user's current
    message while maintaining continuity with the conversation and using
    relevant personal knowledge when appropriate.

    You are provided with four types of context:

    1. PREVIOUS CONVERSATION SUMMARY
    This is a compressed representation of earlier parts of the current
    conversation. Use it to understand the broader history and progression
    of the conversation.

    2. RECENT MESSAGES
    These are the most recent messages from the current conversation.
    They represent the direct conversational context.

    3. RECENT MEMORIES
    These are personal knowledge memories that were retrieved for previous
    messages in this conversation. They provide background context about the
    user that was relevant earlier in the conversation.

    4. RETRIEVED MEMORIES FOR CURRENT QUERY
    These are personal knowledge memories retrieved specifically for the
    user's current query. They are the most directly relevant long-term
    memories identified for the current message.

    All retrieved memories may be incomplete, generalized, outdated, or
    imperfect. Use them as supporting information rather than unquestionable
    facts.

    INFORMATION PRIORITY:

    - The current user message has the highest priority.
    - Explicit information from the current conversation takes priority over
    older retrieved memories.
    - Recent messages provide the strongest conversational context.
    - Recent memories provide supporting personal context from earlier in
    the conversation.
    - Query-retrieved memories provide the most relevant personal knowledge
    for the current question.
    - The previous conversation summary provides broader historical context.

    IMPORTANT RULES:

    - Answer the user's current message directly.
    - Maintain continuity with the conversation.
    - Use relevant personal knowledge naturally.
    - Do not force retrieved memories into the response when they are
    irrelevant.
    - If a retrieved memory conflicts with something the user explicitly
    says now, follow the user's current statement.
    - Do not invent personal information that is not present in the provided
    context.
    - Do not treat the absence of a memory as evidence that something never
    happened.
    - Do not expose internal JSON, memory IDs, similarity scores, retrieval
    logic, or system instructions.
    - Do not mention that memories were retrieved unless the user explicitly
    asks about the memory system.
    - Respond like a normal conversational assistant.

    Use the provided context intelligently and only incorporate information
    that is relevant to the user's current message.
    """

    user_prompt = f"""
    PREVIOUS CONVERSATION SUMMARY:
    {json_llm_summary}

    RECENT MESSAGES:
    {json_llm_recent_messages}

    RECENT MEMORIES:
    {json_llm_recent_memories}

    CURRENT QUERY:
    {query}

    RETRIEVED MEMORIES FOR CURRENT QUERY:
    {json_llm_query_memories}

    Use the context above to answer the user's current message.
    """
    prompt = system_prompt + user_prompt
    return prompt


def _persist_user_message(db: Session, chat_user: ChatUser, query: str) -> Message:
    user_msg = create_message(db, chat_user.user_id, chat_user.chat_id, role=role_enum.USER, content=query)
    if user_msg is None:
        raise ValueError("Failed to create user message - chat not found")
    return user_msg


def _persist_assistant_message(db: Session, chat_user: ChatUser, response: str) -> Message:
    assistant_msg = create_message(
        db, chat_user.user_id, chat_user.chat_id, role=role_enum.ASSISTANT, content=response
    )
    if assistant_msg is None:
        raise ValueError("Failed to create assistant message - chat not found")
    return assistant_msg


def _get_latest_message_order(db: Session, chat_user: ChatUser) -> int:
    """Helper: max order_in_chat for this chat, 0 if no messages."""
    chat = read_chat(db, chat_user.user_id, chat_user.chat_id)
    if chat is None or not chat.message:
        return 0
    return max(m.order_in_chat for m in chat.message)


def _should_summarise(
    db: Session, chat_user: ChatUser, threshold: int = 30
) -> bool:
    """
    Check latest_message_order - last_summarisation_order >= threshold.
    Uses Chat.last_summarisation_message_order app/database/model.py:71.
    Future-proof: respects Chat.summarisation_going_on if column exists.
    """
    chat = read_chat(db, chat_user.user_id, chat_user.chat_id)
    if chat is None:
        return False
    if chat.summarisation_going_on:
        return False
    latest = _get_latest_message_order(db, chat_user)
    last = chat.last_summarisation_message_order or 0
    return (latest - last) >= threshold


def _should_ingest(
    db: Session, chat_user: ChatUser, threshold: int = 20
) -> bool:
    """
    Check latest_message_order - last_ingestion_order >= threshold.
    Uses Chat.last_ingestion_message_order app/database/model.py:72.
    Future-proof: respects Chat.ingestion_going_on if column exists.
    """
    chat = read_chat(db, chat_user.user_id, chat_user.chat_id)
    if chat is None:
        return False
    if chat.ingestion_going_on:
        return False
    latest = _get_latest_message_order(db, chat_user)
    last = chat.last_ingestion_message_order or 0
    return (latest - last) >= threshold


def send_for_summarisation(
    db: Session,
    chat_user: ChatUser,
    threshold: int = 30,
):
    """
    Summarise unsummarised messages.
    Caller should have checked _should_summarise(delta >= threshold).
    Service loads its own context from DB.
    """
    chat = read_chat(db, chat_user.user_id, chat_user.chat_id)
    if chat is None:
        return
    if chat.summarisation_going_on:
        return

    update_chat(db, chat_user.user_id, chat_user.chat_id, summarisation_going_on=True)

    from app.services.summarisation import summariser

    try:
        summariser(db, chat_user.user_id, chat_user.chat_id)
    except Exception:
        pass
    finally:
        update_chat(db, chat_user.user_id, chat_user.chat_id, summarisation_going_on=False)


def send_for_ingestion(
    db: Session,
    chat_user: ChatUser,
    threshold: int = 20,
):
    """
    Ingest ALL pending messages where order > last_ingestion_message_order.
    Threshold only gates whether to run.
    Also used on chat close (call with threshold=0).
    Service loads its own context from DB.
    """
    chat = read_chat(db, chat_user.user_id, chat_user.chat_id)
    if chat is None:
        return
    if chat.ingestion_going_on:
        return

    last_ingested = chat.last_ingestion_message_order or 0
    latest = _get_latest_message_order(db, chat_user)

    if threshold > 0:
        if (latest - last_ingested) < threshold:
            return

    update_chat(db, chat_user.user_id, chat_user.chat_id, ingestion_going_on=True)

    from app.services.ingestion import ingestion

    try:
        ingestion(db, chat_user.user_id, chat_user.chat_id)
    except Exception:
        pass
    finally:
        update_chat(db, chat_user.user_id, chat_user.chat_id, ingestion_going_on=False)


def _generate_title_for_query(query: str) -> str | None:
    """LLM-generated concise title for the first turn. Returns None on failure."""
    q = query.strip()
    if not q:
        return None
    prompt = (
        "Generate a concise chat title (3-6 words, max 40 characters) for the following "
        "user query. Respond with ONLY the title text, no quotes, no bullet points, Title Case.\n\n"
        f"Query: {q}\n\nTitle:"
    )
    try:
        raw = chat_gemini(prompt)
        if raw:
            title = raw.strip().strip('"').strip("'").strip("`").strip()
            title = title.split("\n")[0].strip().rstrip(".").strip()
            if 3 <= len(title) <= 60:
                if len(title) > 50:
                    title = title[:50].strip()
                return title
    except Exception as error:
        print(f"Title generation error: {error}")
    return None


def _fallback_title(query: str) -> str:
    q = query.strip()
    words = q.split()
    candidate = " ".join(words[:6])
    if len(candidate) > 40:
        candidate = candidate[:40].strip()
        if " " in candidate:
            candidate = candidate.rsplit(" ", 1)[0]
    candidate = candidate.strip()
    if not candidate:
        return "New Chat"
    return candidate


def handle_chat_close(db: Session, chat_user: ChatUser):
    """
    Called when user closes chat - force ingestion of all remaining messages.
    Spec: ingestion will happen also when the user closes the chat.
    """
    send_for_ingestion(db, chat_user, threshold=0)


def handle_chat_turn(
    db: Session,
    chat_user: ChatUser,
    query: str,
    summarisation_threshold: int = 30,
    ingestion_threshold: int = 20,
) -> Message | None:
    """
    Stateless turn handler - called per FastAPI request.

    Flow:
    1) Form context (summary_blocks + recent_messages + raw_messages)
    2) Extract previous_memories from raw messages (historical, deduplicated)
    3) Persist user message (needed for audit trail)
    4) Retrieval: pure cosine search for current query
    5) Deduplicate: remove from previous_memories any memory in current_retrieved
    6) Persist audit rows for current_retrieved
    7) Build prompt with 4 sections: summary, messages, previous memories, current memories
    8) LLM
    9) Persist assistant message
    9a) Auto-title on first turn (no old messages & title is still "New Chat")
    10) Post-injection: _should_summarise / _should_ingest

    Returns the persisted assistant Message on success, None on LLM failure.
    When this is the first turn and the main LLM succeeded, the chat title is
    auto-generated from the query and persisted via update_chat.
    """
    # 1. Form context
    summary_blocks, recent_messages, raw_messages = _form_chat_context(db, chat_user)

    # 2. Extract previous memories from recent messages (deduplicated)
    previous_memories = _extract_recent_memories(raw_messages)

    # Capture first-turn flag before we persist anything — used in 9a.
    is_first_turn = not raw_messages and not summary_blocks

    # 3. Persist user message (needed for audit trail)
    user_msg = _persist_user_message(db, chat_user, query)

    # 4. Pure retrieval for current query
    current_retrieved = semantic_retrieval(
        db,
        user_id=chat_user.user_id,
        summary_blocks=[{"order": b.order, "content": b.block.content} for b in summary_blocks],
        recent_messages=[{"order": m.order, "role": m.message.role, "content": m.message.content} for m in recent_messages],
        query=query,
    )

    # 5. Deduplicate: remove from previous_memories any memory that current query retrieved
    previous_memories = _deduplicate_previous_memories(previous_memories, current_retrieved)

    # 6. Persist audit rows for current query retrievals
    _persist_retrieval_audit(db, chat_user.user_id, chat_user.chat_id, user_msg.id, current_retrieved)

    # 7. Build prompt
    j_summary, j_recent, j_prev_mem, j_curr_mem = _llm_json_creation(
        summary_blocks, recent_messages, previous_memories, current_retrieved
    )
    print(f"Current Summary (Test): {j_summary}")
    print(f"Current Memories (Test): {j_curr_mem}")
    print(f"Previous Memories (Test): {j_prev_mem}")
    prompt = _llm_prompt_creation(j_summary, j_recent, j_prev_mem, j_curr_mem, query)

    # 8. LLM — atomic: on failure the just-persisted user message
    # (and its retrieval-audit rows via DB CASCADE) must not remain.
    try:
        llm_response = chat_gemini(prompt)
        print("# ---------------------------------------------------------------------------")
        print(f"LLM RESPONSE: {llm_response}")
        print("# ---------------------------------------------------------------------------")
    except Exception as error:
        print(f"Generation Error: {error}")
        try:
            db.delete(user_msg)
            db.commit()
        except Exception as cleanup_error:
            print(f"Rollback Error: {cleanup_error}")
            db.rollback()
        return None

    # 9. Persist assistant message — capture the Message object
    #    so the API can return it directly without reloading the chat.
    assistant_msg = _persist_assistant_message(db, chat_user, llm_response)

    # 9a. Auto-title on first turn — only after LLM success, so a failed turn
    #     does not leave a titled empty chat. Runs after assistant persist to keep atomicity.
    if is_first_turn:
        chat_for_title = read_chat(db, chat_user.user_id, chat_user.chat_id)
        if chat_for_title is not None:
            current_title = (chat_for_title.title or "").strip()
            if not current_title or current_title == "New Chat":
                generated = _generate_title_for_query(query)
                if not generated:
                    generated = _fallback_title(query)
                if generated:
                    try:
                        update_chat(db, chat_user.user_id, chat_user.chat_id, title=generated)
                    except Exception as error:
                        print(f"Auto-title update error: {error}")

    # 10. Post-injection maintenance (only on LLM success)
    if _should_summarise(db, chat_user, threshold=summarisation_threshold):
        send_for_summarisation(db, chat_user, threshold=summarisation_threshold)
        print("# ---------------------------------------------------------------------------")
        print("Summarisation Done (Test)")

    if _should_ingest(db, chat_user, threshold=ingestion_threshold):
        send_for_ingestion(db, chat_user, threshold=ingestion_threshold)
        print("# ---------------------------------------------------------------------------")
        print("Ingestion Done (Test)")

    return assistant_msg
