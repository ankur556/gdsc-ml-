import os
import openai
import sounddevice as sd
import wavio
import numpy as np
import streamlit as st
import PyPDF2
import speech_recognition as sr
from docx import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# --- Streamlit UI Setup ---
st.title("💬 RAG Chatbot with Voice & Text (No ChromaDB)")

# --- OpenAI API Key ---
openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    openai.api_key = openai_api_key

# --- Document Processing Functions ---
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_pdf_file(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def read_docx_file(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_documents(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".txt":
                texts.append(read_text_file(file_path))
            elif ext == ".pdf":
                texts.append(read_pdf_file(file_path))
            elif ext == ".docx":
                texts.append(read_docx_file(file_path))
    return texts

# --- FAISS Vector Store Setup ---
def create_faiss_index(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text("\n".join(texts))
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# --- Load and Index Documents ---
folder_path = "documents"  # Change this to the path of your document folder
if os.path.exists(folder_path):
    docs = load_documents(folder_path)
    if docs:
        vector_store = create_faiss_index(docs)
        retriever = vector_store.as_retriever()
        st.success("Documents processed successfully!")
    else:
        st.warning("No valid documents found in the folder.")
else:
    st.warning(f"Folder '{folder_path}' not found. Please create it and add some documents.")

# --- Query Function ---
def get_response(query):
    if 'retriever' not in globals():
        return "No documents indexed yet."
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-3.5-turbo", api_key=openai_api_key),
        retriever=retriever
    )
    return qa_chain.run(query)

# --- Streamlit Chat Interface ---
if openai_api_key:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = get_response(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Voice Input without PyAudio ---
def record_audio(filename="input.wav", duration=5, sample_rate=44100):
    st.write("🎤 Recording... Please speak.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype=np.int16)
    sd.wait()
    wavio.write(filename, recording, sample_rate, sampwidth=2)
    st.write("✅ Recording complete!")

def transcribe_audio(filename="input.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError:
        return "Speech recognition service unavailable."

if st.button("🎙 Speak"):
    record_audio()
    transcribed_text = transcribe_audio()
    st.write(f"🗣 You said: {transcribed_text}")

    if transcribed_text.lower() != "could not understand audio.":
        st.session_state.messages.append({"role": "user", "content": transcribed_text})
        with st.chat_message("user"):
            st.markdown(transcribed_text)

        response = get_response(transcribed_text)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
