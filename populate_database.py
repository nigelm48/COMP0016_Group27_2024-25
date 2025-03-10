import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding import embedding_function
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def add_documents_to_chroma(folder_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store by loading files from the given folder
    documents = load_documents_from_directory(folder_path)
    if not documents:
        print("No supported documents found in the directory.")
        return
        
    print(f"Loaded {len(documents)} documents")
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents_from_directory(directory_path):
    """Load documents from various file types in the given directory."""
    all_documents = []
    
    # Get all files in the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            try:
                if file_extension == '.pdf':
                    # For PDFs, we need the directory, not individual files
                    pass
                elif file_extension in ['.md', '.markdown']:
                    print(f"Loading Markdown: {file_path}")
                    all_documents.extend(load_md(file_path))
                elif file_extension in ['.doc', '.docx']:
                    print(f"Loading Word Document: {file_path}")
                    all_documents.extend(load_doc(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Handle PDFs separately since PyPDFDirectoryLoader processes the entire directory
    pdf_documents = load_pdf(directory_path)
    if pdf_documents:
        print(f"Loaded PDFs from directory")
        all_documents.extend(pdf_documents)
        
    return all_documents

def load_pdf(path):
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()

def load_md(path):
    document_loader = UnstructuredMarkdownLoader(path)
    return document_loader.load()

def load_doc(path):
    document_loader = UnstructuredWordDocumentLoader(path)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

