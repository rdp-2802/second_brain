import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")




def llm_deepseek_call(messages: list[dict], HF_TOKEN: str):
    client = InferenceClient(api_key = HF_TOKEN, provider="novita")
    answer = client.chat.completions.create(model = "deepseek-ai/DeepSeek-R1", messages = messages)

    return answer

