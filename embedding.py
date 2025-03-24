# Import the entire transformers module
import transformers
import torch
import os

# Create aliases for needed classes
AutoModelForCausalLM = transformers.AutoModelForCausalLM
AutoTokenizer = transformers.AutoTokenizer
AutoModel = transformers.AutoModel

class LocalHuggingFaceEmbeddings:
    def __init__(self, model_path: str):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch processing to reduce overhead
        batch_size = 1  # Adjust based on your GPU/CPU memory
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,  # Add padding
                max_length=512
            ).to(self.device)
            
            # Generate embeddings in batch
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  
                batch_embeddings = hidden_states.mean(dim=1).tolist()
                embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
  
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  
            embedding = hidden_states.mean(dim=1).squeeze().tolist()
        return embedding


def embedding_function():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "multilingual-e5-small"
    model_path = os.path.join(base_dir, model_path)
    return LocalHuggingFaceEmbeddings(model_path)
