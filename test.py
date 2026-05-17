from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Read API key
api_key = os.getenv("GROQ_API_KEY")

print("API KEY:", api_key[:15], "...")

# Create Groq client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Send request
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ],
    temperature=0
)

# Print answer
print("\n🤖 RESPONSE:\n")
print(response.choices[0].message.content)