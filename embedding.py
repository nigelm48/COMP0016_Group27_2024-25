from langchain_ollama import OllamaEmbeddings

def embedding_function():
    embeddings = OllamaEmbeddings(model="llama3.2:latest")
    return embeddings
