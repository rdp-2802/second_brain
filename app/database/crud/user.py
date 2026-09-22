from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.model import User, SessionLocal, get_db
from uuid import UUID
from sqlalchemy import text as _text
from app.auth.pass_util import hash_password

# class User(Base):
#     __tablename__ = "user"
#     id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     name: Mapped[str] = mapped_column(String(30), nullable=False)
#     email: Mapped[str] = mapped_column(String(30), nullable=False)
#     mobile_no: Mapped[int] = mapped_column(nullable=False)
#     created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

#     chat: Mapped[List["Chat"]] = relationship(back_populates="user")
#     message: Mapped[List["Message"]] = relationship(back_populates="user")
#     memory_detail: Mapped[List["MemoryDetail"]] = relationship(back_populates="user")
#     memory_summary: Mapped[List["MemorySummary"]] = relationship(back_populates="user")
#     retrieved_summary: Mapped[List["RetrievedSummary"]] = relationship(back_populates="user")
#     retrieved_detail: Mapped[List["RetrievedDetail"]] = relationship(back_populates="user")

def create_user(db: Session, name: str, email: str, mobile_no: int, password: str):
    hashed_password = hash_password(password)
    user = User(name = name, email = email, mobile_no = mobile_no, password = hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)

    return user

def read_user(db: Session, id: UUID):
    statement = select(User).where(User.id == id)
    user = db.execute(statement).scalar_one_or_none()

    return user

def read_user_by_email(db: Session, email: str) -> User:
    statement = select(User).where(User.email == email)
    user = db.execute(statement).scalar_one_or_none()

    return user

def read_user_by_mobile_no(db: Session, mobile_no: str) -> User:
    statement = select(User).where(User.mobile_no == mobile_no)
    user = db.execute(statement).scalar_one_or_none()

    return user

def update_user(db: Session, id: UUID, name: str | None, email: str | None, mobile_no: int | None):
    user = read_user(db, id)

    if user is None:
        return None

    if name is not None:
        user.name = name

    if email is not None:
        user.email = email

    if mobile_no is not None:
        user.mobile_no = mobile_no

    db.commit()
    db.refresh(user)

    return user

def delete_user(db: Session, id: UUID):
    user = read_user(db, id)
    if user is None:
        return None
    # Use raw SQL so DB-level CASCADE handles children without ORM nullify issues
    db.execute(_text("DELETE FROM \"user\" WHERE id = :uid"), {"uid": str(id)})
    db.commit()
    return user

def delete_all_users(db: Session):
    statement = "TRUNCATE TABLE \"user\" CASCADE"
    db.execute(_text(statement))
    db.commit()
