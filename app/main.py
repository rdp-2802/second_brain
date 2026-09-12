from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class user_text(BaseModel):
    id: int
    text: str

@app.get('/')
def home():
    return {"message":"bhai ki pehli fast api"}

@app.post('/addtext')
def input_text(text: user_text):
    return {"id":text.id, "text":text.text}