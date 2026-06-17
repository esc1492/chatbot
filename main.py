from fastapi import FastAPI, Request
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

history = {}

SYSTEM_PROMPT = """
너는 영어 학습 비서야.
사용자의 질문에 친절하게 답해줘.
"""

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    user_id = body["userRequest"]["user"]["id"]
    user_msg = body["userRequest"]["utterance"]

    if user_id not in history:
        history[user_id] = []
    history[user_id].append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model="deepseek-chat",  # DeepSeek V4 (Flash)
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[user_id]
        ],
        max_tokens=500
    )
    reply = response.choices[0].message.content

    history[user_id].append({"role": "assistant", "content": reply})

    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": reply}}]
        }
    }
