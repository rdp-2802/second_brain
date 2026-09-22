from sqlalchemy.orm import Session
from app.database.model import User 
from app.database.crud.user import read_user_by_email, read_user_by_mobile_no
from app.database.crud.session import create_session
from app.auth.pass_util import verify_password
from fastapi import HTTPException
from app.auth.schema import LoginRequest

def user_check(db: Session, details: LoginRequest,):

    if details.email is not None:
        user = read_user_by_email(db, details.email)
    elif details.mobile_no is not None:
        user = read_user_by_mobile_no(db, details.mobile_no)
    else:
        return None

    if user is None:
        raise HTTPException(status_code=401, detail="User doesn't exist")

    if verify_password(details.password, user.password):
        return user
    else:
        raise HTTPException(status_code=401, detail="Incorrect Password")
    

def login(db: Session, details: LoginRequest,):

    try:
        user = user_check(db, details)
    except HTTPException as error:
        raise error

    session = create_session(db, user.id)

    return session