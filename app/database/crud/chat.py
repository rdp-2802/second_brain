from sqlalchemy.orm import Session
from sqlalchemy import select
from app.database.model import Chats
from app.database.crud.user import get_user
from datetime import datetime
from uuid import UUID

def add_chat(db: Session, user_id : UUID):
    user = get_user(db, user_id)

    if user is None:
        return None

    chat = Chats(user_id = user_id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

def get_chat(db: Session, user_id: UUID, chat_id: UUID):
    statement = select(Chats).where(Chats.id == chat_id & Chats.user_id == user_id)
    chat = db.execute(statement).scalar_one_or_none()
    return chat

def delete_chat(db: Session, chat_id: UUID):
    chat = get_chat(db, chat_id)
    if chat is None: 
      return None
    db.delete(chat)
    db.commit()
    return chat

def update_chat(db: Session, chat_id: UUID, title: str | None = None, last_message_time: datetime | None = None, last_summary_message_order: int| None, last_memory_extracted_message_order: int | None):
    chat = get_chat(db, chat_id)
    
    if chat is None: 
      return None
    
    if last_message_time != None:
        chat.last_message_at = last_message_time

    if title != None:
        chat.title = title
    
    if last_memory_extracted_message_order != None:
        chat.last_memory_extracted_message_order = last_memory_extracted_message_order

    if last_summary_message_order != None:
        chat.last_summary_message_order = last_summary_message_order

    db.commit()
    db.refresh(chat)
    return chat
