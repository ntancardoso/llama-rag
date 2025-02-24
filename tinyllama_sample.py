from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# replace with model path
model_path = "z:/models/TinyLlama-1.1B-Chat-v1.0"
model2_path = "z:/models/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

embedding_model = SentenceTransformer(model2_path)

texts = ["Product A is a high-quality gadget.", "Product B is a durable gadget."]
embeddings = embedding_model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

query = "Tell me about Product A."
query_embedding = embedding_model.encode([query])
distances, indices = index.search(query_embedding, k=1)
retrieved_text = texts[indices[0][0]]

input_text = f"Based on the following information, answer the user's question: {retrieved_text}\n\nQuestion: {query}\nAnswer:"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=500,
    early_stopping=True,
    num_beams=2,
    no_repeat_ngram_size=2,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)