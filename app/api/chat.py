from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from app.api.auth import validate_user
from app.database.crud.chat import (
    create_chat,
    delete_chat,
    read_chat,
    read_chats,
    update_chat,
)
from app.database.model import get_db
from app.services.chat_orchestration import (
    ChatUser,
    handle_chat_close,
    handle_chat_turn,
)


chat_router = APIRouter()


# ---------------------------------------------------------------------------
# Request models — frontend shape is only: chat id, chat title, messages.
# No role in send-message body: orchestration always persists USER first.
# ---------------------------------------------------------------------------


class ChatCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatTitleUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Response shaping helpers
# ---------------------------------------------------------------------------


def _role_str(role) -> str:
    return role.value if hasattr(role, "value") else str(role)


def _message_dict(message) -> dict:
    return {
        "order": message.order_in_chat,
        "role": _role_str(message.role),
        "content": message.content,
    }


def _sorted_messages(chat) -> list[dict]:
    messages = sorted(list(chat.message or []), key=lambda m: m.order_in_chat)
    return [_message_dict(m) for m in messages]


# ---------------------------------------------------------------------------
# 1. POST /chat/ — create chat, optional title
# ---------------------------------------------------------------------------


@chat_router.post("")
def api_create_chat(
    details: ChatCreateRequest,
    db: DBSession = Depends(get_db),
    user_id: UUID = Depends(validate_user),
):
    chat = create_chat(db, user_id, title=details.title)

    if chat is None:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "chat_id": str(chat.id),
        "title": chat.title,
        "messages": [],
    }


# ---------------------------------------------------------------------------
# 2. GET /chat/ — list user's chats (id + title + last_message_at for sorting)
# ---------------------------------------------------------------------------


@chat_router.get("")
def api_list_chats(
    db: DBSession = Depends(get_db),
    user_id: UUID = Depends(validate_user),
):
    chats = read_chats(db, user_id)

    return {
        "chats": [
            {
                "chat_id": str(chat.id),
                "title": chat.title,
                "last_message_at": chat.last_message_at.isoformat() if chat.last_message_at else None,
            }
            for chat in chats
        ]
    }


# ---------------------------------------------------------------------------
# 3. GET /chat/{chat_id} — full chat: id + title + messages
# ---------------------------------------------------------------------------


@chat_router.get("/{chat_id}")
def api_load_chat(
    chat_id: UUID,
    db: DBSession = Depends(get_db),
    user_id: UUID = Depends(validate_user),
):
    chat = read_chat(db, user_id, chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {
        "chat_id": str(chat.id),
        "title": chat.title,
        "messages": _sorted_messages(chat),
    }


# ---------------------------------------------------------------------------
# 4. PUT /chat/{chat_id} — title-only update (internal flags hidden from frontend)
# ---------------------------------------------------------------------------


@chat_router.put("/{chat_id}")
def api_rename_chat(
    chat_id: UUID,
    details: ChatTitleUpdateRequest,
    db: DBSession = Depends(get_db),
    user_id: UUID = Depends(validate_user),
):
    title = details.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title must not be empty")

    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    updated = update_chat(db, user_id, chat_id, title=title)
    if updated is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {
        "chat_id": str(updated.id),
        "title": updated.title,
    }


# ---------------------------------------------------------------------------
# 5. DELETE /chat/{chat_id} — force ingestion first, then cascading delete
# ---------------------------------------------------------------------------


@chat_router.delete("/{chat_id}")
def api_delete_chat(
    chat_id: UUID,
    db: DBSession = Depends(get_db),
    user_id: UUID = Depends(validate_user),
):
    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Force-ingest remaining messages so memory isn't lost on delete.
    # Ingestion swallows its own errors internally; never block the delete.
    try:
        handle_chat_close(db, ChatUser(user_id=user_id, chat_id=chat_id))
    except Exception as error:
        print(f"Pre-delete ingestion warning: {error}")

    deleted = delete_chat(db, user_id, chat_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {"message": "chat deleted"}


# ---------------------------------------------------------------------------
# 6. POST /chat/{chat_id}/messages — user turn via orchestration.
#    Role is always USER; response is {chat_id, title, messages:[new assistant]}.
# ---------------------------------------------------------------------------


@chat_router.post("/{chat_id}/messages")
def api_send_message(
    chat_id: UUID,
    details: SendMessageRequest,
    db: DBSession = Depends(get_db),
    user_id: UUID = Depends(validate_user),
):
    content = details.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Message must not be empty")

    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat_user = ChatUser(user_id=user_id, chat_id=chat_id)

    try:
        assistant_msg = handle_chat_turn(db, chat_user, query=content)
    except ValueError:
        raise HTTPException(status_code=404, detail="Chat not found")

    if assistant_msg is None:
        # Orchestration already rolled back the user message on LLM failure.
        raise HTTPException(status_code=502, detail="LLM failed to respond")

    # Title may have been auto-generated on first turn inside handle_chat_turn.
    # Re-read so the response carries the fresh title; falls back to pre-turn title.
    fresh = read_chat(db, user_id, chat_id)
    final_title = fresh.title if fresh and fresh.title else chat.title

    return {
        "chat_id": str(chat_id),
        "title": final_title,
        "messages": [_message_dict(assistant_msg)],
    }


# ---------------------------------------------------------------------------
# 7. GET /chat/{chat_id}/messages — kept for completeness (frontend unused)
# ---------------------------------------------------------------------------


@chat_router.get("/{chat_id}/messages")
def api_list_messages(
    chat_id: UUID,
    db: DBSession = Depends(get_db),
    user_id: UUID = Depends(validate_user),
):
    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {
        "chat_id": str(chat.id),
        "title": chat.title,
        "messages": _sorted_messages(chat),
    }
