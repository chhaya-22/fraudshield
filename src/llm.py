import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile", json_mode: bool = False) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content
