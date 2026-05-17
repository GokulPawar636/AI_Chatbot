from query import advanced_retrieval
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

# Create Groq client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# ---------------- CONTEXT ----------------

def build_context(results):

    context = ""

    for i, (doc, score) in enumerate(results):

        meta = doc.metadata

        context += f"""
========================
DOCUMENT {i+1}

Similarity Score:
{score}

Building:
{meta.get("building")}

Category:
{meta.get("head")}

Serial Number:
{meta.get("sr_no")}

Built-up Area:
{meta.get("built_rate")}

Saleable Area:
{meta.get("saleable_rate")}

Amount:
{meta.get("amount")}

Remarks:
{meta.get("remarks")}

Basic Rate:
{meta.get("basic_rate")}

Section:
{meta.get("section")}

Original Content:
{doc.page_content}

========================
"""

    return context.strip()



# ---------------- MAIN ----------------
def generate_answer(query):

    # 🔍 Retrieve relevant documents
    results = advanced_retrieval(query)

    if not results:
        return "❌ No relevant construction data found."

    # 🔥 Build context from retrieved docs
    context = build_context(results)

    # 🔥 Smart AI Prompt

    
    prompt = f"""
You are an advanced AI construction assistant.

Your task is to answer user questions naturally using ONLY the provided project data.

IMPORTANT RULES:

1. NEVER invent values.
2. Use ONLY relevant records.
3. Give professional and human-friendly answers.
4. Format numbers properly with commas.
5. Mention units clearly.
6. Keep responses concise and readable.
7. Do NOT mention document numbers.
8. Do NOT say "according to document".
9. If data is unavailable, clearly say so.
10. Understand user intent naturally.
11. Amount value in indian rupees should be formatted with commas and "INR" suffix, e.g. "1,23,45,678 INR".
12. Built-up and saleable area should be formatted with commas and "SFT" suffix, e.g. "1,23,456 SFT".

-----------------------------------
USER QUESTION:
{query}
-----------------------------------

PROJECT DATA:
{context}

-----------------------------------

RESPONSE FORMAT RULES:

- Start directly with the answer.
- Use clean formatting.
- Use bullet points if needed.
- Explain technical values simply.
- Highlight important values using markdown.
- Keep response natural like ChatGPT.

GOOD RESPONSE EXAMPLE:

The built-up area for "A Building" is "154,236.71 square feet (SFT)".

⚪ Details
- Building: A Building
- Category: Built-up Area
- Area: 154,236.71 SFT

🏗️ Explanation
This represents the total constructed floor area of the building, including all usable structural sections.

Now generate the best possible response.
"""
    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional construction project AI assistant. " 
                        "Always provide structured, professional, concise, and human-friendly responses. " 
                        "Use markdown formatting for readability."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=400,
            top_p=0.9
        )

        answer = response.choices[0].message.content.strip()

        return answer

    except Exception as e:

        return f"❌ Error generating response: {str(e)}"
# ---------------- INTERACTIVE ----------------
if __name__ == "__main__":

    print("🚀 System Ready (FAISS + LLM loaded once)")

    while True:

        query = input("\nAsk question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = generate_answer(query)

        print("\n🤖 FINAL ANSWER:\n")
        print(answer)