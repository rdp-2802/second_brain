import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.5-flash-lite"


def chat_gemini(content: str) -> str:
    chat = client.chats.create(model = MODEL)
    try:
        response = chat.send_message(content)
        return response.text
    except ClientError as error:
        raise
