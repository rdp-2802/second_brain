from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select, text as _text
from datetime import datetime, timedelta, timezone

from app.database.model import Session, User
from app.database.crud.user import read_user
import secrets


def session_token() -> str:
    return secrets.token_urlsafe(32)

def create_session(
    db: Session,
    user_id: UUID,
):
    user = read_user(db, user_id)
    if user is None:
        return None

    SESSION_DURATION = timedelta(days = 30)
    session_id = session_token()

    valid_till = datetime.now(timezone.utc) + SESSION_DURATION

    session = Session(
        id = session_id,
        user_id = user_id,
        valid_till = valid_till,
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def read_session(
    db: Session,
    user_id: UUID,
    session_id: str
):
    user = read_user(db, user_id)
    if user is None:
        return None

    statement = select(Session).where(
        Session.id == session_id,
        Session.user_id == user_id
    )

    session = db.execute(statement).scalar_one_or_none()

    return session


def read_session_by_id(
    db: Session,
    session_id: str
):
    statement = select(Session).where(Session.id == session_id)
    session = db.execute(statement).scalar_one_or_none()
    return session


def read_sessions(
    db: Session,
    user_id: UUID
):
    user = read_user(db, user_id)
    if user is None:
        return []

    statement = select(Session).where(Session.user_id == user_id)
    sessions = db.execute(statement).scalars().all()
    return sessions


def update_session(
    db: Session,
    user_id: UUID,
    session_id: str,
    valid_till: datetime | None = None
):
    user = read_user(db, user_id)
    if user is None:
        return None

    session = read_session(db, user_id, session_id)
    if session is None:
        return None

    if valid_till is not None:
        session.valid_till = valid_till

    db.commit()
    db.refresh(session)
    return session


def delete_session(
    db: Session,
    user_id: UUID,
    session_id: str
):
    user = read_user(db, user_id)
    if user is None:
        return None

    session = read_session(db, user_id, session_id)
    if session is None:
        return None

    db.delete(session)
    db.commit()
    return session


def delete_all_sessions(
    db: Session,
    user_id: UUID
):
    user = read_user(db, user_id)
    if user is None:
        return None

    statement = _text("DELETE FROM session WHERE user_id = :uid")
    db.execute(statement, {"uid": str(user_id)})
    db.commit()
    return user


def delete_expired_sessions(
    db: Session
):
    now = datetime.now()
    statement = _text("DELETE FROM session WHERE valid_till < :now")
    result = db.execute(statement, {"now": now})
    db.commit()
    return result.rowcount