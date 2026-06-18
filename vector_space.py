from sentence_transformers import SentenceTransformer, util
import torch

print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

node_text = "Server Room"
query_text = "Where did the murder take place?"

# converting texts to vectors
print("\nConverting texts to 384-dimensional vectors...")
node_embedding = model.encode(node_text, convert_to_tensor=True)
query_embedding = model.encode(query_text, convert_to_tensor=True)

# 4. Calculate the Cosine Similarity between the two vectors
cosine_score = util.cos_sim(node_embedding, query_embedding)

print("-" * 40)
print(f"Graph Node: '{node_text}'")
print(f"User Query: '{query_text}'")
print(f"Semantic Proximity (Cosine Score): {cosine_score.item():.4f}")
print("-" * 40)