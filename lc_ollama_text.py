from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma, SKLearnVectorStore
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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

db = Chroma.from_texts(documents, OllamaEmbeddings(model="llama3.2"))

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)


# result = retriever.invoke("Who livs in a pineaple under the sea?")
# print(result)


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)
# print(custom_rag_prompt)

question = "Who livs in a pineaple under the sea?"
retriever_context = retriever.invoke(question)

augmented_query = custom_rag_prompt.format(context=retriever_context, question=question)

context_content = retriever_context[0].page_content if retriever_context else "No context found."

augmented_query = custom_rag_prompt.format(context=context_content, question=question)

llm = ChatOllama(model="llama3.2")
llm_output = llm.invoke(augmented_query)

final_output = StrOutputParser().invoke(llm_output)

print(final_output)