import os
import streamlit as st
import PyPDF2
import docx
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import uuid

# Initialize OpenAI client
client = OpenAI()

# Set up Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot with ChromaDB & OpenAI")

# API Key Input
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = api_key if api_key else ""

# Upload Files
uploaded_files = st.sidebar.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="Chromadb")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="documents_collection", embedding_function=sentence_transformer_ef)

def read_text_file(file):
    return file.read().decode("utf-8")

def read_pdf_file(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx_file(file):
    doc = docx.Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_document(file):
    if file.type == "text/plain":
        return read_text_file(file)
    elif file.type == "application/pdf":
        return read_pdf_file(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx_file(file)
    return ""

def process_document(file):
    content = read_document(file)
    return content

if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            content = process_document(file)
            chunks = content.split(". ")
            ids = [f"{file.name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file.name, "chunk": i} for i in range(len(chunks))]
            collection.add(documents=chunks, metadatas=metadatas, ids=ids)
            st.success(f"{file.name} added to knowledge base.")

query = st.text_input("Ask a question:")

if st.button("Search") and query:
    results = collection.query(query_texts=[query], n_results=3)
    context = "\n\n".join(results['documents'][0])
    response = client.chat.completions.create(
        model="gpt-4", 
        messages=[{"role": "user", "content": context + "\n\n" + query}],
        temperature=0, max_tokens=500
    ).choices[0].message.content
    st.subheader("Answer:")
    st.write(response)
    st.subheader("Sources:")
    for meta in results['metadatas'][0]:
        st.write(f"{meta['source']} (Chunk {meta['chunk']})")
