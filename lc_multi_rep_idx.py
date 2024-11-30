# multi representation indexing
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
from langchain_core.documents import Document


loaders = [
    TextLoader("blog.langchain.dev_announcing-langsmith_.txt", encoding='utf-8'),
    TextLoader("blog.langchain.dev_automating-web-research_.txt", encoding='utf-8'),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOllama(model="llama3.2", max_retries=0)
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 3})

vectorstore = Chroma(collection_name="summaries", embedding_function=OllamaEmbeddings(model="llama3.2"), persist_directory="./chroma_db/mri_data")

store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))


query = "What is LangSmith?"
sub_docs = vectorstore.similarity_search(query)
sub_docs[0]

retrieved_docs = retriever.invoke(query)

# retrieved_docs[0].page_content[0:500]

# len(retrieved_docs[0].page_content)

print(retrieved_docs)