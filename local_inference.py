from huggingface_hub import hf_hub_download
from llama_cpp import Llama

print("Downloading Llama-3 Model...")

model_path = hf_hub_download(
    repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
)

# initializing the model with Llama
llm = Llama(
    model_path=model_path,
    n_ctx=2048, # maximum number of tokens the model can read at once
    n_threads=4
)

# testing prompt for the model
prompt = "Question: As a detective, explain why finding a secret letter from a rival company is suspicious in a murder case. Keep it under 3 sentences."

output = llm(
    f"<|system|>\nYou are a smart AI detective.\n<|user|>\n{prompt}\n<|assistant|>",
    max_tokens=150,
    temperature=0.1, # Reduce hallucination margin
    echo=False
)

print("\nLlama-3's Response:")
print(output['choices'][0]['text'])
print("-" * 50)