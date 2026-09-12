from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-embedding-2"

client = genai.Client(api_key = api_key)

def generate_embedding(text: str):
    embedding = client.models.embed_content(model = MODEL_ID, contents = text, config=types.EmbedContentConfig(output_dimensionality=1024))
    result = embedding.embeddings[0].values
    return result

