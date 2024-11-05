from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, SKLearnVectorStore


documents = [
    "Python is a high-level programming language recognized for its readability and versatile libraries.",
    "Java is a widely used programming language suitable for building large-scale applications.",
    "JavaScript plays a crucial role in web development, allowing for interactive web experiences.",
    "Machine learning is a branch of artificial intelligence focused on training algorithms to make predictions.",
    "Deep learning, a field within machine learning, employs neural networks to capture complex patterns in data.",
    "Landmarks often hold cultural and historical significance in various regions.",
    "Many museums house extensive collections of artworks and artifacts.",
    "Spongebob Squarepants lives in a pineapple under the sea.",
    "Artificial intelligence encompasses machine learning techniques that enable systems to learn from data."
]

db = Chroma.from_texts(documents, OllamaEmbeddings(model="llama3.1"))

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)


result = retriever.invoke("Who livs in a pineaple under the sea?")
print(result)