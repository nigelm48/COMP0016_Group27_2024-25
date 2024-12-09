from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalHuggingFaceEmbeddings:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  
                embedding = hidden_states.mean(dim=1).squeeze().tolist()
                embeddings.append(embedding)
        return embeddings


def embedding_function():
    model_path = "llama3.2-1b"  
    return LocalHuggingFaceEmbeddings(model_path)