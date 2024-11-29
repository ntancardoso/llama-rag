import streamlit as st
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF

st.set_page_config(page_title='Sample RAG application')
st.title('Sample RAG Application')

#def get_prompt(identifier):
#    prompts = {
#        "rag": "Given the context: {context}, answer the question: {question}",
#        "summary": "Summarize the following: {context}"
#    }
#    return prompts.get(identifier, "No prompt found.")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def read_document(file):
    if file.type == "application/pdf":
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    else:
        return file.read().decode()

def generate_response(uploaded_files, query_text):
    documents = []
    for uploaded_file in uploaded_files:
        document_text = read_document(uploaded_file)
    documents.append(document_text)
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = []
    for document in documents:
        texts.extend(text_splitter.create_documents([document]))

    llm = ChatOllama(model="llama3.2")
    # Select embeddings
    embeddings =  OllamaEmbeddings(model="llama3.2")

    # Create a vectorstore from documents
    database = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db/data_db")
    # Create retriever interface
    retriever = database.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Create QA chain
    response = rag_chain.invoke(query_text)
    return response

uploaded_files = st.file_uploader('Upload one or more articles', type=['txt', 'pdf'], accept_multiple_files=True)
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_files)

result = None

with st.form(key='rag_form', clear_on_submit=False, border=False):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_files, query_text)
            result = response

if result:
    st.info(result)