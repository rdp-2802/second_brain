import json
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.model import Messages
from app.database.crud.message_block import (
    get_message_blocks_by_chat,
    add_message_block,
    update_message_block,
)
from app.database.crud.message_join_block import add_message_join_block
from app.services.llm import chat_gemini

def last_message_finder(messages: list["Messages"]):
    messages.sort(key = lambda x: x.order_in_chat)
    len = len(messages)
    last_summary_message_order = messages[len-1].order_in_chat

def _build_summary_context(
    db: Session,
    chat_id: UUID,
    messages: list[Messages],
):
    """Fetch and shape the data needed for the block summary prompt.
    Pure data assembly — no prompt text lives here."""

    messages_and_retrieved_context = []

    for message in messages:
        retrieved_summaries = [
            retrieved_summary.knowledge_summary.summary
            for retrieved_summary in message.retrieved_summaries
            if retrieved_summary.knowledge_summary is not None
        ]

        retrieved_details = [
            retrieved_detail.knowledge_detail.detail_content
            for retrieved_detail in message.retrieved_details
            if retrieved_detail.knowledge_detail is not None
        ]

        messages_and_retrieved_context.append({
            "role": message.role,
            "content": message.content,
            "retrieved_context": {
                "summaries": retrieved_summaries,
                "details": retrieved_details,
            },
        })

    blocks = get_message_blocks_by_chat(db, chat_id)

    previous_block_summary = [
        {
            "order": block.order_in_chat,
            "summary": block.summary_text,
        }
        for block in blocks
        if block.summary_text is not None
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


def block_summary_prompt(
    db: Session,
    chat_id: UUID,
    user_id: UUID,
    messages: list[Messages],
) -> str:
    messages_and_retrieved_context, previous_block_summary = _build_summary_context(
        db, chat_id, messages
    )

    return _render_summary_prompt(
        user_id,
        chat_id,
        messages_and_retrieved_context,
        previous_block_summary,
    )

def block_summariser(db: Session, messages: list["Messages"], chat_user: chat_user)

    chat_id = chat_user.chat_id
    user_id = chat_user.user_id

    prompt = block_summary_prompt(block, messages)

    summary = chat_gemini(prompt)

    block = add_message_block(db, chat_id, summary)

    last_summary_message_order = last_message_finder(messages)

    update_chat(db, last_summary_message_order = last_summary_message_order)

    for message in messages:
        add_message_join_block(db, chat_id, block.id, message.id)

    return block