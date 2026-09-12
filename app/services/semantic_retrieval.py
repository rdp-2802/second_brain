from app.database.model import KnowledgeSummary, KnowledgeDetail
from app.database.crud import knowledge_summary, knowledge_detail
from app.database.crud.knowledge_detail import get_knowledge_details_by_summary
from uuid import UUID
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Literal
from app.database.model import Messages, MessageBlock

class previous_memory(BaseModel):
    summary_id: UUID
    summary_content: str
    details: list[str]

class summary_block(BaseModel):
    order: int
    block: MessageBlock

class recent_messages(BaseModel):
    order: int
    message: Messages
    retrieved_summary: list[KnowledgeSummary]
    retrieved_detail: list[KnowledgeDetail]

class retrieved_memory(BaseModel):
    memory_type: Literal["summary", "detail"]
    memory: KnowledgeDetail | KnowledgeSummary
    similarity: float

#INPUT: summary blocks, recent messages + retrieved knowledge, query
# step 1: create summary string from summary blocks
# step 2: create recent messages + retrieved knowledge string from recent messages list
# step 3: create unified summary + recent + query message strings for classification
# step 4: create query embedding for similarity search
# step 5: get top k semantic details + summaries
# step 6: run classification for figuring out the needed details + summaries
# step 7: remove the retrieved knowledge from previous messages which are getting repeated in the needed details + summaries
# step 8: return the retrieved needed details + recent_messages object after removing repeated memories 

# later improvements: allow dynamic retrieval instead of top k semantic details + summaries (v2-v3 headache not now)

def _create_block_summary_str(summaries: list[summary_block]):

    summaries.sort(key= lambda x: x.order)
    block_summary_str = ""

    for summary in summaries:
        block_summary_str += summary.block.summary_text + '\n'

    return block_summary_str

def _create_recent_message_str(recent_messages: list[recent_messages]):

    recent_messages.sort(key = lambda x: x.order)
    recent_messages_str = ""

    for message in recent_messages:
        recent_messages_str += message.message.role + ": " + message.message.content + "\n"

        retrieved_summary_str = "Retrieved Summaries: "
        retrieved_detail_str = "Retrieved Details: "

        for summary in message.retrieved_summary:
            retrieved_summary_str += summary.content + "\n"
        for detail in message.retrieved_detail:
            retrieved_detail_str += detail.detail_content + "\n"

        recent_messages_str += retrieved_summary_str + retrieved_detail_str

    return recent_messages_str

def _create_conversation_str(block_summary_str: str, recent_message_str: str, query: str):

    conversation_str = "Previous Conversation Summary: " + block_summary_str + "Recent Messages: " + recent_message_str + "Query: " + query

    return conversation_str

def top_k_retrieval(db: Session, user_id: UUID, query_embedding: list[float], k: int, cosine_lim: float):

    k_summary = knowledge_summary.get_near_embedding(db, user_id, query_embedding, k)
    k_detail = knowledge_detail.get_near_embedding(db, user_id, query_embedding, k)

    k_memory = []
    memory = []

    for summary, distance in k_summary:
        similarity = 1 - distance
        if similarity >= cosine_lim:
            memory.append(retrieved_memory(
                                memory_type="summary",
                                memory=summary,
                                similarity= similarity,
                        ))
    for detail, distance in k_detail:
        similarity = 1 - distance
        if similarity >= cosine_lim:
            memory.append(retrieved_memory(
                                memory_type="detail",
                                memory=detail,
                                similarity= similarity,
                        ))

    memory.sort(key =lambda x: x.similarity, reverse=True)

    retrieved = memory[:k]

    return retrieved

def classification_retrieved_needed(retrieved: list[retrieved_memory], conversation_str: str):
    retrieved_needed = retrieved

    return retrieved_needed

def update_retrieved_summary_detail_db(retrieved: list[retrieved_memory], message: Messages):
    

def _recreate_recent_messages(recent_messages: list[recent_messages], retrieved_needed: list[retrieved_memory]):

    retrieved_summary_ids = {
        item.memory.id
        for item in retrieved_needed
        if item.memory_type == "summary"
    }

    retrieved_detail_ids = {
        item.memory.id
        for item in retrieved_needed
        if item.memory_type == "detail"
    }

    for message in recent_messages:

        message.retrieved_summary = [
            summary
            for summary in message.retrieved_summary
            if summary.id not in retrieved_summary_ids
        ]

        message.retrieved_detail = [
            detail
            for detail in message.retrieved_detail
            if detail.id not in retrieved_detail_ids
        ]               

    return recent_messages

def semantic_retrieval(db: Session, chat_user: chat_user, summaries: list[summary_block], recent_messages: list[recent_messages], query: str):
    user_id = chat_user.user_id
    
    block_summary_str = _create_block_summary_str(summaries)
    recent_message_str = _create_recent_message_str(recent_messages)
    conversation_str = _create_conversation_str(block_summary_str, recent_message_str, query)

    query_embedding = generate_embedding(query)
    retrieved = top_k_retrieval(db, user_id, query_embedding, k = 5, cosine_lim= 0.5)

    retrieved_needed = classification_retrieved_needed(retrieved, conversation_str)

    recent_messages = _recreate_recent_messages(recent_messages, retrieved_needed)

    return retrieved_needed, recent_messages



def retrieve_memory(db:Session, user_id: UUID, embedding: list[float], summary_limit:int, detail_limit: int):
    summaries = get_near_embeddings(db, user_id, summary_limit, embedding)
    summaries = [summary for summary, score in summaries]

    memory = []

    for summary in summaries:
        details = get_knowledge_details_by_summary(db, summary.id, detail_limit)
        details_content = []
        for detail in details:
            details_content.append(detail.detail_content)
        memory.append(previous_memory(summary_id = summary.id, summary_content = summary.content, details = details_content))

    return memory