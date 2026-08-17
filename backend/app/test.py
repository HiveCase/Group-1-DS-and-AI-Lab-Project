from openai import OpenAI
from core.config import get_settings
client = OpenAI(
    api_key=get_settings().groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

for model in client.models.list().data:
    print("Available Groq model: %s", model.id)