import json
from datetime import datetime
from pydantic import BaseModel, TypeAdapter
from uuid import UUID
from app.database.model import role_enum, User, Chats, Messages
from app.database.crud.user import get_user
from app.database.crud.knowledge_summary import add_knowledge_summary, update_knowledge_summary
from app.database.crud.knowledge_detail import add_knowledge_detail
from sqlalchemy.orm import Session
from app.models.embedding import generate_embedding
from app.services.retrieval import retrieve_memory, previous_memory
from app.models.llm import chat_gemini

# UPGRADE: Give Chat Summary with recent messages for better context understanding and memory generation

class chat_user(BaseModel):
    user_id: UUID
    chat_id: UUID

class ingestion_message(BaseModel):
    id: UUID
    role: role_enum
    content: str

class output_detail(BaseModel):
    detail_content: str
    source_message_ids: list[UUID]

class output_memory(BaseModel):
    summary_id: UUID | None
    summary_content: str
    details: list[output_detail]

def ingestion_message_converter(messages: list["Messages"]):
    ingestion_messages = []
    for message in messages:
        ingestion_message = ingestion_message(id = message.id, role = message.role, content = message.content)
        ingestion_messages.append(ingestion_message)

    return ingestion_messages

def last_message_finder(messages: list["Messages"]):
    messages.sort(key = lambda x: x.order_in_chat)
    len = len(messages)
    last_memory_extracted_message_order = messages[len-1].order_in_chat


def ingestion_message_embedding(ingestion_messages: list[ingestion_message]):
    message_string = "\n".join(message.content for message in ingestion_messages)
    embedding = generate_embedding(message_string)

    return embedding

def ingestion_prompt(user_name: str, ingestion_messages: list[ingestion_message], previous_memories: list[previous_memory]):
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

    prompt_1 = f"""
    # Memory Ingestion

    ## Brief

    You are a Memory Ingestion LLM for a personal knowledge base.

    Extract durable and useful NEW knowledge about the user from the current
    conversation. First create the Knowledge Details, then decide whether those
    Details belong to an existing Knowledge Summary or require a new Summary.

    A Summary represents a broad, persistent cluster of related knowledge.
    A Detail represents specific new knowledge and evidence from the current
    conversation.

    Do not summarize the conversation. Store only knowledge that can be useful
    for understanding the user in future conversations. Most conversations
    contain nothing worth storing — returning [] is the expected outcome far
    more often than not.

    ---

    ## Input

    ### User (user_name)
    {user_name}

    ### Current Date-Time (datetime)
    {current_datetime}

    Use this when interpreting relative time expressions such as "today",
    "yesterday", "last week", etc.

    ### Current Conversation
    {conversation_json}

    Each message contains:
    - id: message UUID
    - role: role_enum(User, Agent, System)
    - content: message content

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

    ## Example A — new Summary, existing Summary update, and a supersession

    ### User
    Rishi

    ### Current Date-Time
    11:39:10 AM 26th February 2026 

    ### Previous Similar Memories

    memories_json = [
        {
            "summary_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "summary_content": "The user has been building a personal RAG-based knowledge management tool in Python with PostgreSQL and pgvector. He chose PyMuPDF for parsing, RecursiveCharacterTextSplitter for chunking, sentence-transformers plus Cohere reranking for retrieval, and was targeting a bottom-up build with explicit input/output contracts between stages.",
            "details": [
                "The user said he prefers building bottom-up, defining each stage's input/output contract before implementation, rather than designing the whole pipeline top-down first."
            ]
        }
    ]

    ### Current Conversation

    conversation_json = [
        {
            "id": "11111111-1111-4111-8111-111111111111",
            "role": "user",
            "content": "Update — I dropped Cohere reranking today while coding, it wasn't worth the API cost for this project size. Going with a simpler cross-encoder reranker instead."
        },
        {
            "id": "22222222-2222-4222-8222-222222222222",
            "role": "assistant",
            "content": "Makes sense, cross-encoders are a solid lighter-weight option."
        },
        {
            "id": "33333333-3333-4333-8333-333333333333",
            "role": "user",
            "content": "lol yeah anyway I'm starving, gonna go grab lunch"
        }
    ]

    ### Expected Processing

    Detail 1: The user replaced Cohere reranking with a local cross-encoder
    reranker, citing API cost relative to project size — this changes a
    previously stated architecture decision. Source: message 1.

    The "gonna go grab lunch" message is casual and temporary — it produces no
    Detail.

    Deciding placement: Detail 1 updates the existing second-brain-rag Summary,
    and the update should show the change (Cohere → cross-encoder) rather than
    just dropping the old fact silently.

    ### Example Output

    [
        {
            "summary_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "summary_content": "The user has been building a personal RAG-based knowledge management tool in Python with PostgreSQL and pgvector. He chose PyMuPDF for parsing and RecursiveCharacterTextSplitter for chunking. He originally planned Cohere reranking but later replaced it with a local cross-encoder reranker due to API cost relative to the project's size. He prefers building bottom-up, defining each stage's input/output contract before implementation.",
            "details": [
                {
                    "detail_content": "The user replaced Cohere reranking with a local cross-encoder reranker in his RAG pipeline, citing that the API cost wasn't justified for the project's size.",
                    "source_message_ids": ["11111111-1111-4111-8111-111111111111"]
                }
            ]
        }
    ]

    ---

    ## Example B — nothing worth storing

    ### User
    Rishi

    ### Current Date-Time
    11:39:10 AM 26th February 2026

    ### Previous Similar Memories
    memories_json = []

    ### Current Conversation

    conversation_json = [
        {
            "id": "44444444-4444-4444-8444-444444444444",
            "role": "user",
            "content": "can you explain what a hash collision is"
        },
        {
            "id": "55555555-5555-4555-8555-555555555555",
            "role": "assistant",
            "content": "A hash collision is when two different inputs produce the same hash output..."
        },
        {
            "id": "66666666-6666-4666-8666-666666666666",
            "role": "user",
            "content": "ah got it, thanks"
        }
    ]

    ### Expected Processing

    This is a one-off factual question with no durable information about the
    user — no preference, decision, event, or change is expressed. Nothing
    qualifies as a Detail.

    ### Example Output

    []

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

def response_to_pydantic(response: str):
    adapter = TypeAdapter(list[output_memory])
    memories = adapter.validate_json(response)
    return memories

def db_memory_ingestion(db: Session, memories: list[output_memory], user_id: UUID, chat_id: UUID):
    for memory in memories:
        if memory.summary_id is None:
            updated_memory = add_knowledge_summary(db, memory.summary_content, user_id)
        else:
            updated_memory = update_knowledge_summary(db, user_id, memory.summary_id, memory.summary_content)
        for detail in memory.details:
            add_knowledge_detail(db, updated_memory.summary_id, chat_id, detail.source_message_ids, detail.detail_content, user_id)

def ingestion(db: Session, messages: list["Messages"], ingestion_user: chat_user):

    ingestion_messages = ingestion_message_converter(messages)

    embedding = ingestion_message_embedding(ingestion_messages)

    user_id = ingestion_user.user_id
    chat_id = ingestion_user.chat_id

    summary_limit = 5
    detail_limit = 5    

    previous_memories = retrieve_memory(db, user_id, embedding, summary_limit, detail_limit)

    user = get_user(db, user_id)
    user_name = user.name

    prompt = ingestion_prompt(user_name, ingestion_messages, previous_memories)

    response = chat_gemini(prompt)

    memories = response_to_pydantic(response)

    db_memory_ingestion(db, memories, user_id, chat_id)

    last_memory_extracted_message_order = last_message_finder(messages)

    update_chat(db, last_memory_extracted_message_order):
