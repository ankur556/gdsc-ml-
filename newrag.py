import os
os.system("pip install -r requirements1.txt")
import openai
import docx
import PyPDF2
import speech_recognition as sr
import streamlit as st

# --- Setup ---
st.title("💬 Document-Based Chatbot with Voice & Text")
st.write("This chatbot can help you search and process documents, as well as take voice or text inputs.")

# --- File Upload ---
uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["txt", "pdf", "docx"])

document_chunks = []

# Helper function to read different document formats
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

def read_document(file, file_extension):
    if file_extension == "txt":
        return read_text_file(file)
    elif file_extension == "pdf":
        return read_pdf_file(file)
    elif file_extension == "docx":
        return read_docx_file(file)
    return ""

# Helper function to split document into chunks
def split_text(text, chunk_size=500):
    sentences = text.replace('\n', ' ').split('. ')
    chunks, current_chunk, current_size = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith('.'):
            sentence += '.'
        sentence_size = len(sentence)

        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# --- Process Uploaded Files ---
if uploaded_files:
    document_chunks.clear()
    for file in uploaded_files:
        file_extension = file.name.split('.')[-1].lower()
        content = read_document(file, file_extension)
        chunks = split_text(content)
        document_chunks.extend(chunks)
    st.success(f"Successfully uploaded {len(uploaded_files)} document(s)!")

# --- Simple Search ---
def simple_search(query, n_results=2):
    return [chunk for chunk in document_chunks if query.lower() in chunk.lower()][:n_results]

def get_context_with_sources(results):
    return "\n\n".join(results) if results else "No relevant documents found."

# --- OpenAI Response Generation ---
def generate_response(query, context):
    prompt = f"""Based on the following context, provide a relevant response. If no relevant info is found, say so.
    
    Context:
    {context}
    
    User: {query}
    Assistant:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )

    return response["choices"][0]["message"]["content"]

# --- Streamlit Chat Interface ---
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to proceed.", icon="🔑")
else:
    openai.api_key = openai_api_key  # Set the API key

    # Initialize session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the input, search for relevant documents, and get the response
        results = simple_search(prompt)
        context = get_context_with_sources(results)
        response = generate_response(prompt, context)

        with st.chat_message("assistant"):
            st.markdown(response)

        # Store assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Upload and process audio files for speech-to-text
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if audio_file:
        recognizer = sr.Recognizer()
        audio = sr.AudioFile(audio_file)
        
        try:
            with audio as source:
                st.write("Processing audio... Please wait.")
                audio_data = recognizer.record(source)
                voice_input = recognizer.recognize_google(audio_data)
                st.write(f"You said: {voice_input}")

                # Process the voice input, search for relevant documents, and get the response
                st.session_state.messages.append({"role": "user", "content": voice_input})
                with st.chat_message("user"):
                    st.markdown(voice_input)

                results = simple_search(voice_input)
                context = get_context_with_sources(results)
                response = generate_response(voice_input, context)

                with st.chat_message("assistant"):
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand your speech. Please try again.")
        except sr.RequestError as e:
            st.write(f"Error with the speech recognition service: {e}")
