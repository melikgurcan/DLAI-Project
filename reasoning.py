import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

with open('data/mystery_graph.json', 'r', encoding='utf-8') as f:
    triplets = json.load(f)

context_lines = []
for item in triplets:
    context_lines.append(f"- {item['subject']} {item['relation']} {item['object']}")

graph_context = "\n".join(context_lines)

system_prompt = """You are an expert detective and reasoning AI. You are provided with relationships from a Knowledge Graph regarding a locked-room murder mystery.
Your task is to analyze ONLY these connections, use step-by-step logical deduction (Chain-of-Thought), and determine who the most likely culprit is.
Explain your reasoning clearly, explicitly mentioning the connections you used from the provided graph."""

user_question = f"Here is the evidence (Graph Connections):\n{graph_context}\n\nQuestion: Based on these connections, who is the killer and how did you deduce this?"

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ],
    model="llama-3.1-8b-instant",
    temperature=0.1
)

print(response.choices[0].message.content)