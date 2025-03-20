import argparse
import os
import shutil
# Import complete modules instead of specific classes
import langchain_community.document_loaders as document_loaders
import langchain_text_splitters
import langchain.schema.document
from embedding import embedding_function
import langchain_community.vectorstores as vectorstores

# Create aliases for frequently used classes
PyPDFDirectoryLoader = document_loaders.PyPDFDirectoryLoader
PyPDFLoader = document_loaders.PyPDFLoader
RecursiveCharacterTextSplitter = langchain_text_splitters.RecursiveCharacterTextSplitter
Document = langchain.schema.document.Document
Chroma = vectorstores.Chroma
UnstructuredMarkdownLoader = document_loaders.UnstructuredMarkdownLoader
UnstructuredWordDocumentLoader = document_loaders.UnstructuredWordDocumentLoader

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")

def add_documents_to_chroma(file_or_folder_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args, _ = parser.parse_known_args()  # Avoid GUI argparse errors

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Determine if it's a file or a folder
    if os.path.isfile(file_or_folder_path):
        print(f"📂 Processing single file: {file_or_folder_path}")
        documents = load_single_file(file_or_folder_path)
    elif os.path.isdir(file_or_folder_path):
        print(f"📁 Processing folder: {file_or_folder_path}")
        documents = load_documents_from_directory(file_or_folder_path)
    else:
        print(f"Invalid path: {file_or_folder_path}")
        return

    if not documents:
        print(f"No content extracted from {file_or_folder_path}")
        return

    print(f"Loaded {len(documents)} document(s)")
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_single_file(file_path):
    """Load a single Markdown, Word, or PDF file."""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            return load_pdf(file_path)
        elif ext in [".md", ".markdown"]:
            return load_md(file_path)
        elif ext in [".doc", ".docx"]:
            return load_doc(file_path)
        else:
            print(f" Unsupported file type: {ext}")
            return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []



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
    """Load PDF content from a file path or directory."""
    if os.path.isfile(path):
        document_loader = PyPDFLoader(path)
        return document_loader.load()
    else:
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
        print(f" Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print(" No new documents to add")

    db._client._system.stop()
    db=None

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

