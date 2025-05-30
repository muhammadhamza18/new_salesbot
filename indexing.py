# import os
# import re
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load_dotenv()

# def extract_pdfs_from_data_folder(data_folder="data"):
#     """Extract all PDF files from the specified data folder."""
#     pdf_files = []
#     for root, dirs, files in os.walk(data_folder):
#         for file in files:
#             if file.endswith(".pdf"):
#                 pdf_files.append(os.path.join(root, file))
#     return pdf_files

# def extract_text_from_pdf(pdf_path):
#     """Extract and clean text from a PDF file."""
#     text = ""
#     with open(pdf_path, "rb") as file:
#         reader = PdfReader(file)
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text += page.extract_text() if page.extract_text() else ""

#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def chunk_pdf_text(text, chunk_size=1000, overlap=200):
#     """Chunk the given text into chunks using RecursiveCharacterTextSplitter."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=overlap,
#         separators=["\n\n", "\n", " ", ""],  
#         length_function=len
#     )
    
#     chunks = text_splitter.split_text(text)
#     return chunks

# def generate_embeddings_for_chunks(chunks):
#     """Generate embeddings for each chunk of text."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
#     chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]
#     return chunk_embeddings

# pdf_files = extract_pdfs_from_data_folder()
# print(f"Found {len(pdf_files)} PDF files.")

# for pdf_file in pdf_files:
#     text = extract_text_from_pdf(pdf_file)
#     chunks = chunk_pdf_text(text)
#     embeddings = generate_embeddings_for_chunks(chunks)
#     print(f"PDF: {pdf_file}, Chunks: {len(chunks)}, Embeddings: {len(embeddings)}")

import os
import re
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

def extract_pdfs_from_data_folder(data_folder="data"):
    """Extract all PDF files from the specified data folder."""
    pdf_files = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def extract_text_from_pdf(pdf_path):
    """Extract and clean text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_pdf_text(text, chunk_size=1000, overlap=200):
    """Chunk the given text into manageable pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],  
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings_for_chunks(chunks, embeddings):
    """Generate embeddings for each text chunk."""
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]
    return chunk_embeddings

pdf_files = extract_pdfs_from_data_folder()
print(f"Found {len(pdf_files)} PDF files.")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize ChromaDB
persist_directory = "./chrom_rag_one_langchain_db"
os.makedirs(persist_directory, exist_ok=True)

db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

for pdf_file in pdf_files:
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_pdf_text(text)
    chunk_embeddings = generate_embeddings_for_chunks(chunks, embeddings)
    
    # Create a unique collection name
    collection_name = os.path.splitext(os.path.basename(pdf_file))[0]
    
    # Store chunks and embeddings in Chroma
    db.add_texts(texts=chunks, metadatas=[{"source": pdf_file}] * len(chunks))
    
    print(f"PDF: {pdf_file}, Chunks: {len(chunks)}, Collection: {collection_name}")

# Persist the database
db.persist()
print("Embedding process completed successfully!")