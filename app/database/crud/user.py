from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.model import User
from uuid import UUID

def add_user(db: Session, name: str, email: str, mobile_no: int):
    user = User(name = name, email = email, mobile_no = mobile_no)
    db.add(user)
    db.commit()
    db.refresh(user)

    return user

def get_user(db: Session, id: UUID):
    statement = select(User).where(User.id == id)
    user = db.execute(statement).scalar_one_or_none()

    return user

def get_all_user(db: Session):
    statement = select(User)
    users = db.execute(statement).scalars().all()

    return users

def delete_user(db: Session, id: UUID):
    user = get_user(db, id)
    if user is None:
        return None
    else:
        db.delete(user)
        db.commit()
        return user

def update_user(db: Session, id: UUID, name: str, email: str, mobile_no: int):
    user = get_user(db, id)

    if user is None:
        return None

    user.name = name
    user.email = email
    user.mobile_no = mobile_no

    db.commit()
    db.refresh(user)

    return user