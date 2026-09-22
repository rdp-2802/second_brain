from fastapi import FastAPI
from fastapi import APIRouter
from pydantic import BaseModel

from app.api.auth import auth_router
from app.api.chat import chat_router

app = FastAPI()

app.include_router(auth_router, prefix="/auth")
app.include_router(chat_router, prefix="/chat")


@app.get('/')
def home():
    return {"message":"bhai ki pehli fast api"}

