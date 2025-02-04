import os
import shutil
import openai
import chromadb
from docx import Document
import PyPDF2
import speech_recognition as sr
import streamlit as st
from chromadb.utils import embedding_functions

# Ensure ChromaDB Works Without SQLite Upgrade
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3  # Force ChromaDB to use the right SQLite

# Delete old ChromaDB storage
chromadb_path = "Chromadb"
if os.path.exists(chromadb_path):
    shutil.rmtree(chromadb_path)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=chromadb_path)

# Streamlit UI Setup
st.title("💬 Document-Based Chatbot with Voice & Text")
st.write("This chatbot can search documents and take voice or text inputs.")

# OpenAI API Key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    openai.api_key = openai_api_key

# ChromaDB Collection Setup
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="documents_collection", embedding_function=sentence_transformer_ef)

# Document Processing Functions
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def read_docx_file(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_document(file_path):
    _, ext = os.path.splitext(file_path.lower())
    if ext == '.txt':
        return read_text_file(file_path)
    elif ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def process_document(file_path):
    try:
        content = read_document(file_path)
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []

def add_to_collection(collection, ids, texts, metadatas):
    if texts:
        collection.add(documents=texts, metadatas=metadatas, ids=ids)

# Semantic Search with ChromaDB
def semantic_search(collection, query, n_results=2):
    results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    return results

def get_context_with_sources(results):
    if not results['documents']:
        return "No relevant documents found.", []
    context = "\n\n".join(results['documents'][0])
    sources = [f"{meta['source']} (chunk {meta['chunk']})" for meta in results['metadatas'][0]]
    return context, sources

# OpenAI Chat Response
def generate_response(query, context, conversation_history=""):
    prompt = f"""
    Based on the following context and conversation history, provide a relevant response.
    If the answer cannot be derived from the context, say: "I cannot answer this based on the provided information."
    
    Context from documents:
    {context}
    
    Previous conversation:
    {conversation_history}
    
    Human: {query}
    
    Assistant:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# Streamlit Chat Interface
if openai_api_key:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        results = semantic_search(collection, prompt)
        context, sources = get_context_with_sources(results)
        response = generate_response(prompt, context)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Speak"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source)
            try:
                voice_input = recognizer.recognize_google(audio)
                st.write(f"You said: {voice_input}")
                results = semantic_search(collection, voice_input)
                context, sources = get_context_with_sources(results)
                response = generate_response(voice_input, context)
                with st.chat_message("assistant"):
                    st.markdown(response)
            except sr.UnknownValueError:
                st.write("Sorry, I couldn't understand.")
            except sr.RequestError:
                st.write("Speech recognition error.")
