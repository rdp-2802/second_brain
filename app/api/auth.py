from fastapi import APIRouter, Depends, HTTPException, Response, Cookie
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session as DBSession
from uuid import UUID

from app.database.model import User, Session, get_db
from app.auth.login import login
from app.auth.signup import signup
from app.auth.schema import LoginRequest, SignupRequest
from app.database.crud.session import read_session_by_id
from app.database.crud.user import delete_user


auth_router = APIRouter()


def validate_user(session_id: str | None = Cookie(default = None), db: DBSession = Depends(get_db),):

    if session_id is None:
        raise HTTPException(status_code=401, detail="Not Logged In")

    session = read_session_by_id(db, session_id)

    if session is None:
        raise HTTPException(status_code=401, detail="No Session Found")
    
    if session.valid_till < datetime.now(timezone.utc):
        db.delete(session)
        db.commit()

        raise HTTPException(status_code=401, detail="Session Expired")

    return session.user_id


@auth_router.post("/login")
def api_login(details: LoginRequest, response: Response, db: Session = Depends(get_db),):
    try:
        session = login(db, details,)
    except HTTPException as error:
        raise error

    SESSION_DURATION = timedelta(days = 30)

    response.set_cookie(
        key="session_id",
        value=session.id,
        httponly=True,
        #secure=True,
        samesite="lax",
        max_age=int(SESSION_DURATION.total_seconds()),
    )

    return {"message": "Login Successful"}

@auth_router.post("/signup")
def api_signup(details: SignupRequest, response: Response, db: DBSession = Depends(get_db),):
    session = signup(db, details)

    SESSION_DURATION = timedelta(days = 30)

    response.set_cookie(
        key="session_id",
        value=session.id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=int(SESSION_DURATION.total_seconds()),
    )

    return {"message": "Sign Up Successful"}

@auth_router.post("/logout")
def api_logout(response: Response, session_id: str | None = Cookie(default=None), db: DBSession = Depends(get_db),):
    if session_id is None:
        raise HTTPException(status_code=401, detail="No Session Found")

    session = read_session_by_id(db, session_id)

    if session is None:
        raise HTTPException(status_code=401, detail="No Session Found")

    db.delete(session)
    db.commit()

    response.delete_cookie("session_id")
    return {"message":"Logged Out"}

@auth_router.post("/delete_account")
def api_delete_account(response: Response, session_id: str | None = Cookie(default=None), db: DBSession = Depends(get_db),):
    if session_id is None:
        raise HTTPException(status_code=401, detail="No Session Found")

    session = read_session_by_id(db, session_id)

    if session is None:
        raise HTTPException(status_code=401, detail="No Session Found")

    delete_user(db, session.user_id)

    response.delete_cookie("session_id")
    return {"message": "Account Deleted"}