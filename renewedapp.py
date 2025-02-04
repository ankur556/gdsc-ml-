import os
import shutil
import openai
import chromadb
import PyPDF2
import speech_recognition as sr
import streamlit as st
from chromadb.utils import embedding_functions
from docx import Document

# --- Cleanup Previous ChromaDB Storage ---
chromadb_path = "Chromadb"
if os.path.exists(chromadb_path):
    shutil.rmtree(chromadb_path)  # Delete the folder

# --- Initialize ChromaDB with DuckDB (No SQLite) ---
client = chromadb.PersistentClient(path="Chromadb", database="duckdb")

# --- Streamlit UI Setup ---
st.title("💬 Document-Based Chatbot with Voice & Text")
st.write("This chatbot can search and process documents, as well as take voice or text inputs.")

# --- Set OpenAI API Key ---
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    openai.api_key = openai_api_key  

# --- ChromaDB Collection Setup ---
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="documents_collection", embedding_function=sentence_transformer_ef)

# --- Document Processing Functions ---
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def read_docx_file(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        return read_text_file(file_path)
    elif ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def split_text(text, chunk_size=500):
    sentences = text.replace('\n', ' ').split('. ')
    chunks, current_chunk = [], []
    current_size = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
        sentence += '.' if not sentence.endswith('.') else ''
        if current_size + len(sentence) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_size = [sentence], len(sentence)
        else:
            current_chunk.append(sentence)
            current_size += len(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_document(file_path):
    try:
        content = read_document(file_path)
        chunks = split_text(content)
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []

def add_to_collection(collection, ids, texts, metadatas):
    if texts:
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            collection.add(
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

def process_and_add_documents(folder_path):
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    for file_path in files:
        ids, texts, metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)

# --- Semantic Search with RAG ---
def semantic_search(query, n_results=2):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results

def get_context_with_sources(results):
    if not results['documents']:
        return "No relevant documents found.", []
    
    context = "\n\n".join(results['documents'][0])
    sources = [f"{meta['source']} (chunk {meta['chunk']})" for meta in results['metadatas'][0]]
    return context, sources

# --- OpenAI Response Generation ---
def get_prompt(context, conversation_history, query):
    return f"""Based on the following context and conversation history, please provide a relevant and contextual response.
    If the answer cannot be derived from the context, only use the conversation history or say "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""

def generate_response(query, context, conversation_history=""):
    prompt = get_prompt(context, conversation_history, query)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# --- Streamlit Chat Interface ---
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

        results = semantic_search(prompt)
        context, sources = get_context_with_sources(results)
        response = generate_response(prompt, context)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Speak"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening... Please speak.")
            audio = recognizer.listen(source)
            try:
                voice_input = recognizer.recognize_google(audio)
                st.write(f"You said: {voice_input}")

                st.session_state.messages.append({"role": "user", "content": voice_input})
                with st.chat_message("user"):
                    st.markdown(voice_input)

                results = semantic_search(voice_input)
                context, sources = get_context_with_sources(results)
                response = generate_response(voice_input, context)

                with st.chat_message("assistant"):
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

            except sr.UnknownValueError:
                st.write("Sorry, I could not understand your speech.")
            except sr.RequestError:
                st.write("Sorry, there was an issue with the speech recognition service.")

