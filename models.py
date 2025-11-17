import os
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")

client = OpenAI(api_key=API_KEY)
models = client.models.list()

available_chat_models = [m.id for m in models.data if "chat" in m.id or "gpt" in m.id]
print("\n".join(available_chat_models))
