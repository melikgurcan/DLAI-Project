import os
import networkx as nx
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from groq import Groq

# initializing API
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Layer 1: Loading NER Model...")
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

print("Layer 2: Loading Embedding Model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# raw Data and analysis
story = """ Elias Thorne was found poisoned in the Conservatory. 
Isabella, his niece, was written out of the will yesterday. 
Arthur, the groundskeeper, argued with Elias about the exotic plants. 
Isabella purchased rare toxic seeds from the dark web. 
Arthur was fixing the sprinklers in the Garden during the incident. """

print("\n1: Automated Entity Extraction")
entities = ner_pipeline(story)
extracted_nodes = list(set([ent['word'] for ent in entities]))
print(f"Extracted Entities: {extracted_nodes}")

print("\n2: Semantic Vector Analysis")
target_concept = "Where is the murder location?"
target_emb = embed_model.encode(target_concept, convert_to_tensor=True)

# calculating the proximity of entities to the "murder location" concept
best_match = None
highest_score = 0

for node in extracted_nodes:
    node_emb = embed_model.encode(node, convert_to_tensor=True)
    score = util.cos_sim(target_emb, node_emb).item()
    print(f"Proximity score for '{node}': {score:.4f}")
    if score > highest_score:
        highest_score = score
        best_match = node

print(f"Mathematically most probable murder location: {best_match}")

print("\n3: Reasoning with Llama-3.1")
system_prompt = "You are an AI detective. Use the provided context to explain the murder concisely."
user_prompt = f"Context: {story}\nBased on the text, who is the prime suspect for the murder in the {best_match} and what is their motive?"

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    model="llama-3.1-8b-instant",
    temperature=0.1
)

print(f"\nDetective Llama-3.1's Conclusion:\n{response.choices[0].message.content}")