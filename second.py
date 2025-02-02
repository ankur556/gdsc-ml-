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

