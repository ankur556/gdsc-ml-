import os
os.system("pip install -r requirements.txt")
import docx
import PyPDF2
def read_text_file(file path: str): #read content from a text file given in input by the user
with open(file_path,'r',encoding='utf-8')as file:#The file is opened in read mode ('r') with UTF-8 encoding
  return file.read()
def read_pdf_file(filepath: str): #read content from a pdf file
  text="" #intializing text whare we will store all of our data
  with open(file_path,'rb')as file:
    pdf_reader = PyPDF2.PdfReader(file)#intializing pdf reader using a python library called pypdf2 which is also mentioned in requirements
    for page in pdf_reader.pages:#creating a for loop to keep on adding elements of rhe text into the text variable 
      text+=page.extract_text+\n
      return text #returning the text variable after adding the whole text onto an empty variable
  def read_docx_file(filepath: str):
    doc=docx.Document(file_path)
return "/n".join([pareagraph.text for paragraph in doc.paragraphs])
"""question out of curiosity answered by chatgpt
Why Different Functions for Different File Formats?
Each file type has a different internal structure, requiring a different method for extracting text:

File Type	Why a Different Function?
.txt	Plain text files store raw text, so open(file, 'r') is enough.
.pdf	PDFs are binary files with structured formatting, requiring a specialized library (PyPDF2) to extract text from pages.
.docx	Word files use an XML-based structure, so we need python-docx to correctly extract text from paragraphs.
If we used the same function for all three:

A .pdf file wouldn't be read properly with open(file, 'r') since it's binary.
A .docx file wouldn't return readable text without parsing its XML structure.
Each format requires a specialized approach to correctly extract text."""

def read_document(file_path: str):
    
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.txt':
        return read_text_file(file_path)
    elif file_extension == '.pdf':
        return read_pdf_file(file_path)
    elif file_extension == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")#this bot only supports 3 format of files which are text doc and pdf

def split_text(text: str, chunk_size: int = 500):
    #Split text into chunks while preserving sentence boundaries
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Ensure proper sentence ending ro rnsure it we add fullstop to it
        if not sentence.endswith('.'):
            sentence += '.'

        sentence_size = len(sentence)

        # Check if adding this sentence would exceed chunk size which is 500 in this case which can be changed in thiscase by entering diffrent values
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    # Add the last chunk if it exists ro the array of chunks 
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

import chromadb
from chromadb.utils import embendings_functions

client= chromadb.PersetentClient(path="Chromadb")
setence_transformer_ef=embedding_functions.SentenceTransformerEmbeddingFunction( model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="documents_collection",embedding_function=sentence_transformer_ef)
def process_document(file_path: str):
    """Process a single document and prepare it for ChromaDB"""
    try:
        content = read_document(file_path) # Read the document to get all the data in the doc
        chunks = split_text(content)# Split into chunks so that we can look for the soloution more effecirntly
        file_name = os.path.basename(file_path)# Prepare metadata for ml model to read the inputs properly
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []


def add_to_collection(collection, ids, texts, metadatas):
   # Add documents to collection in batches
    if not texts:
        return

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

def process_and_add_documents(collection, folder_path: str):
    files = [os.path.join(folder_path, file) #Process all documents in a folder and add to collection
             for file in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, file))]

    for file_path in files:
        print(f"Processing {os.path.basename(file_path)}...")
        ids, texts, metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)
        print(f"Added {len(texts)} chunks to collection")
# Initialize ChromaDB collection 
collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=sentence_transformer_ef
)

# Process and add documents from a folder
folder_path = "/docs"
process_and_add_documents(collection, folder_path)

def semantic_search(collection, query: str, n_results: int = 2):
    """Perform semantic search on the collection"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def get_context_with_sources(results):
    #Extract context and source information from search results
    #Combine document chunks into a single context
    context = "\n\n".join(results['documents'][0])

    #Format sources with metadata
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})" 
        for meta in results['metadatas'][0]
    ]

    return context, sources








#setting up open ai
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"#using thare api kry to keep the expense on them

def get_prompt(context: str, conversation_history: str, query: str):
    """Generate a prompt combining context, history, and query"""
    prompt = f"""Based on the following context and conversation history,which i have talked about prior 
    please provide a relevant and contextual response(using the context above). If the answer cannot 
    be derived from the context, only use the conversation history or if you still cant find the answer please respond with 
    "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""

    return prompt

def generate_response(query: str, context: str, conversation_history: str = ""):
    """Generate a response using OpenAI with conversation history"""
    prompt = get_prompt(context, conversation_history, query)

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or gpt-3.5-turbo for lower cost but in thsi case i have dicided to go for gpt-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Lower temperature for more focused responses
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def rag_query(collection, query: str, n_chunks: int = 2):
    """Perform RAG query: retrieve relevant chunks and generate answer"""
    # Get relevant chunks
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)
    # Generate response
    response = generate_response(query, context)
    return response, sources
import uuid
from datetime import datetime
import json

# In-memory conversation store
conversations = {}

def create_session():
    #Create a new conversation session
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id

def format_history_for_prompt(session_id: str, max_messages: int = 5):
    #Format conversation history for inclusion in prompts and to remove any prior peomts with something new you are trying to make
    history = get_conversation_history(session_id, max_messages)
    formatted_history = ""

    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"

    return formatted_history.strip()

def contextualize_query(query: str, conversation_history: str, client: OpenAI):
    #Convert follow-up questions into standalone queries
    contextualize_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone 
    question which can be understood without the chat history. Do NOT answer 
    the question, just reformulate it if needed and otherwise return it as is."""

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": contextualize_prompt},
                {"role": "user", "content": f"Chat history:\n{conversation_history}\n\nQuestion:\n{query}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error contextualizing query: {str(e)}")
        return query  # Fallback to original query

#combining rag components

def get_prompt(context, conversation_history, query):
  prompt = f"""Based on the following context and conversation history, please provide a relevant and contextual response.
    If the answer cannot be derived from the context, only use the conversation history or say "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""
  return prompt


  # Updated generate response function with conversation history also passed for Chatbot Memory
def generate_response(query: str, context: str, conversation_history: str = ""):
    """Generate a response using OpenAI with conversation history"""
    prompt = get_prompt(context, conversation_history, query)
    # print(prompt)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0,
        max_tokens=500
    )

    return response.choices[0].message.content

  
def conversational_rag_query(
    collection,
    query: str,
    session_id: str,
    n_chunks: int = 3
):
    """Perform RAG query with conversation history"""
    # Get conversation history
    conversation_history = format_history_for_prompt(session_id)

    # Handle follo up questions
    query = contextualize_query(query, conversation_history, client)
    print("Contextualized Query:", query)

    # Get relevant chunks
    context, sources = get_context_with_sources(
        semantic_search(collection, query, n_chunks)
    )
    print("Context:", context)
    print("Sources:", sources)


    response = generate_response(query, context, conversation_history)

    # Add to conversation history
    add_message(session_id, "user", query)
    add_message(session_id, "assistant", response)

    return response, sources
