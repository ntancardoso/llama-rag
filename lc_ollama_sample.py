from langchain_ollama import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(model="llama3.1")

embeddings = embeddings_model.embed_documents(
    [
        "This is a fundamental concept in Retrieval-Augmented Generation.",
        "AI-powered online learning is becoming increasingly popular.",
        "Generative AI is a rapidly evolving field.",
        "I am composing this text using my keyboard.",
        "Python is a versatile programming language."
    ]
)

print(len(embeddings))

print(len(embeddings[0]))
