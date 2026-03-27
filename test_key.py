from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)

key = os.getenv("GROK_API_KEY", "")
print("Key length:", len(key))
print("Key prefix:", key[:8])
print("Key suffix:", key[-4:])
print("Base URL:", os.getenv("GROK_BASE_URL"))
print("Model:", os.getenv("GROK_MODEL"))

try:
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "say hi"}]
    )
    print("SUCCESS:", r.choices[0].message.content)
except Exception as e:
    print("FAILED:", e)
