from sqlalchemy.orm import Session
from app.database.model import User 
from app.database.crud.user import create_user
from app.database.crud.session import create_session
from app.auth.schema import SignupRequest


def signup(db: Session, details: SignupRequest):
    name = details.name
    email = details.email
    mobile_no = details.mobile
    password = details.password

    user = create_user(db, name, email, mobile_no, password)
    session = create_session(db, user.id)
    if session is None:
        db.delete(user)
        db.commit()
    return session