import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.6-flash"


def chat_gemini(content: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=content,
    )

    return response.text