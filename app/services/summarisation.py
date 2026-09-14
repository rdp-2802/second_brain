import json
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.model import Message, MessageBlock
from app.database.crud.chat import read_chat, update_chat
from app.database.crud.message import read_messages_after_order
from app.database.crud.message_block import create_message_block, read_message_blocks_by_chat
from app.database.crud.message_join_block import create_message_join_block

try:
    from app.models.llm import chat_gemini
except ImportError:
    from app.services.llm import chat_gemini


def _build_summary_context(
    messages: list[Message],
    blocks: list[MessageBlock],
) -> tuple[list[dict], list[dict]]:
    """
    Build prompt context from raw Message and MessageBlock objects.
    """
    messages_and_retrieved_context = []

    for message in messages:
        # Extract retrieved summaries/details via join table relationships
        retrieved_summaries = []
        for rs in getattr(message, "retrieved_summary", []) or []:
            mem = getattr(rs, "memory_summary", None)
            if mem is not None:
                retrieved_summaries.append(mem.content)

        retrieved_details = []
        for rd in getattr(message, "retrieved_detail", []) or []:
            mem = getattr(rd, "memory_detail", None)
            if mem is not None:
                retrieved_details.append(mem.content)

        messages_and_retrieved_context.append({
            "role": message.role.value if hasattr(message.role, "value") else str(message.role),
            "content": message.content,
            "retrieved_context": {
                "summaries": retrieved_summaries,
                "details": retrieved_details,
            },
        })

    previous_block_summary = [
        {
            "order": block.order_in_chat,
            "summary": block.content,
        }
        for block in blocks
    ]

    return messages_and_retrieved_context, previous_block_summary


def _render_summary_prompt(
    user_id: UUID,
    chat_id: UUID,
    messages_and_retrieved_context: list[dict],
    previous_block_summary: list[dict],
) -> str:

    messages_json = json.dumps(messages_and_retrieved_context, ensure_ascii=False)
    previous_block_json = json.dumps(previous_block_summary, ensure_ascii=False)

    prompt_1 = f"""
    # Message Block Summarization

    ## Brief

    You are a Message Block Summarization LLM for a personal knowledge base.

    Your task is to summarize the current set of conversation messages into a
    compact, coherent summary that preserves the important conversational
    context needed to understand what has happened in this chat.

    The current messages may contain retrieved personal knowledge from the
    knowledge base. Use that retrieved context when it is relevant for
    understanding the conversation, but do not treat retrieved context as
    something that was necessarily said in the current messages.

    Previous Message Block summaries are provided as additional context. Use
    them to maintain continuity and avoid unnecessarily repeating information
    that is already adequately represented.

    The output should summarize the current messages and their relevant
    retrieved context, while maintaining continuity with the previous blocks.

    ---

    ## Input

    ### User ID
    {user_id}

    ### Chat ID
    {chat_id}

    ### Current Messages and Retrieved Context

    {messages_json}

    Each message contains:

    - role: the role of the message
    - content: the message content
    - retrieved_context:
        - summaries: knowledge summaries retrieved for this message
        - details: knowledge details retrieved for this message

    Retrieved context represents information retrieved from the user's
    knowledge base. It is provided to help understand the conversation and
    should be incorporated only when relevant.

    ### Previous Message Block Summaries

    {previous_block_json}

    Each previous block contains:

    - order: the block's position within the chat
    - summary: the summary generated for that block

    Previous block summaries are context only. Do not blindly repeat them.
    Use them to understand the ongoing trajectory of the conversation and to
    maintain continuity.

    ---
    """

    prompt_2 = """
    ## Task

    1. Read all Current Messages and their Retrieved Context.

    2. Produce a single coherent summary of the current messages. Capture the
    important topics, decisions, reasoning, questions, conclusions, plans,
    changes, and other context that would be useful for understanding this
    conversation later.

    3. Use Retrieved Context when it materially helps explain the current
    conversation. Do not dump or mechanically reproduce retrieved knowledge
    that is unrelated to the current messages.

    4. Use Previous Message Block Summaries to maintain continuity with the
    earlier conversation. Avoid repeating information unnecessarily when it is
    already represented there, but include it when it is necessary to make the
    current block understandable.

    5. Preserve important relationships between statements. If the user
    changed their mind, rejected an approach, made a decision, or refined an
    idea, preserve that progression rather than flattening everything into a
    generic summary.

    6. The summary should represent the conversation, not merely list the
    messages. Combine related messages into coherent statements.

    7. Distinguish between information stated by the user, information supplied
    by the assistant, and information retrieved from the knowledge base. Do not
    claim that retrieved knowledge was newly stated by the user in the current
    messages.

    8. Do not invent information, intentions, opinions, events, or conclusions
    that are not supported by the Current Messages, Retrieved Context, or
    Previous Message Block Summaries.

    9. Preserve useful technical details, names, decisions, constraints, and
    terminology when they are relevant to the conversation.

    10. Do not include irrelevant conversational filler, greetings, or trivial
    exchanges unless they contribute meaningful context.

    11. Keep the summary compact. It should contain enough information to
    reconstruct the important state of the conversation without reproducing
    the conversation itself.

    12. Write the summary in the same language predominantly used by the user
    in the Current Messages. Do not translate to English by default.

    ---

    ## Output

    Return ONLY the summary string.

    Do not return JSON.
    Do not return markdown.
    Do not include explanations.
    Do not include labels such as "Summary:".

    Example:

    The user is designing a personal knowledge-base system and is currently
    implementing the conversation memory pipeline. They decided to represent
    conversation history using message blocks and retrieve relevant knowledge
    summaries and details for individual messages. The current discussion
    focuses on constructing the block summarization prompt and correctly
    associating retrieved summaries and details with each message.
    """

    return prompt_1 + prompt_2


def summariser(db: Session, user_id: UUID, chat_id: UUID) -> None:
    """
    Summarise oldest half of unsummarised messages into a MessageBlock.
    Loads its own context from DB. Returns None (stateless).

    Only the oldest half is summarised so the remaining half stays as
    recent messages for LLM context. This also avoids duplicating messages
    in both summary blocks and recent messages.

    1. Sanity check: chat.summarisation_going_on
    2. Load messages after last_summarisation_message_order (ascending)
    3. Take oldest half: batch = messages[:len(messages)//2]
    4. Load ALL message_blocks (previous block summaries)
    5. Build prompt, LLM call
    6. create_message_block + create_message_join_block for batch only
    7. update_chat -> last_summarisation_message_order = max batch order
    """
    # Sanity check
    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        return
    if chat.summarisation_going_on:
        return

    last_summarised = chat.last_summarisation_message_order or 0

    # Load unsummarised messages (ascending by order_in_chat)
    messages = read_messages_after_order(db, chat_id, last_summarised)
    if not messages:
        return

    # Only summarise oldest half — leave the rest as recent messages for context.
    # Guarantees at least 1 message stays unsummarised for the next LLM prompt.
    if len(messages) < 2:
        return
    batch = messages[: len(messages) // 2]
    if not batch:
        return

    # Load ALL blocks (previous block summaries for continuity)
    blocks = read_message_blocks_by_chat(db, user_id, chat_id) or []

    # Build prompt context from batch only
    messages_context, blocks_context = _build_summary_context(batch, blocks)

    prompt = _render_summary_prompt(user_id, chat_id, messages_context, blocks_context)

    # LLM call
    summary_text = chat_gemini(prompt)

    # Create MessageBlock
    block = create_message_block(db, user_id, chat_id, content=summary_text)
    if block is None:
        return

    # Link each batched message to the block
    for message in batch:
        create_message_join_block(db, user_id, chat_id, block.id, message.id)

    # Update last summarisation order to max of batch only
    last_order = max(m.order_in_chat for m in batch)
    update_chat(db, user_id, chat_id, last_summarisation_message_order=last_order)
