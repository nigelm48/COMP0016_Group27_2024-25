import tkinter as tk
from tkinter import scrolledtext, filedialog
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from populate_database import add_documents_to_chroma
from embedding import embedding_function

# Initialize model and tokenizer
model_path = 'LLM_Model/ultra/llama3.2-11b'  # Replace with your model's path
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 for faster inference
).to(device)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    device=0 if torch.cuda.is_available() or torch.mps.is_available() else -1,
    pad_token_id=tokenizer.eos_token_id
)

# Initialize LLM pipeline
llm = generator

# Chroma vector store path
CHROMA_PATH = "chroma"

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to retrieve similar documents
def retrieve_similar_documents(query, top_k=3):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function()
    )
    results = db.similarity_search(query, k=top_k)
    return [result.page_content for result in results]

# Function to generate a response from the model
def generate_response(input_text, context=""):
    prompt_template = PromptTemplate(
        template="{context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=context, question=input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            num_beams=3,
            no_repeat_ngram_size=2,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()
    last_period_index = answer.rfind(".")
    if last_period_index != -1:
        answer = answer[:last_period_index + 1]
    answer = answer.replace("\n", "").replace("\r", "").replace("\t", "")

    return answer

# Function to send query from the GUI
def send_query():
    query = input_box.get("1.0", tk.END).strip()
    if query:
        output_box.insert(tk.END, f"You: {query}\n", "user")
        output_box.see(tk.END)
        similar_docs = retrieve_similar_documents(query)
        context = "\n".join(similar_docs)
        response = generate_response(query, context=context)
        output_box.insert(tk.END, f"AI: {response}\n\n", "bot")
        output_box.see(tk.END)
        input_box.delete("1.0", tk.END)

# Function to browse and process a folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        output_box.insert(tk.END, f"Processing files in folder: {folder_selected}\n", "bot")
        output_box.see(tk.END)
        try:
            # Add documents to the Chroma vector store using populate_database.py
            add_documents_to_chroma(folder_selected)
            output_box.insert(tk.END, "Documents added to the database successfully.\n", "bot")
        except Exception as e:
            output_box.insert(tk.END, f"Error: {str(e)}\n", "bot")
        output_box.see(tk.END)

# Create the main window
root = tk.Tk()
root.title("AI RAG Assistant")

# Output box (Scrollable)
output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=60, state=tk.NORMAL)
output_box.tag_config("user", foreground="blue")
output_box.tag_config("bot", foreground="green")
output_box.pack(pady=10)

# Input box
input_box = tk.Text(root, height=3, width=50)
input_box.pack(pady=5)

# Send button
send_button = tk.Button(root, text="Send", command=send_query)
send_button.pack(pady=5)

# Browse folder button
browse_button = tk.Button(root, text="Load Documents", command=browse_folder)
browse_button.pack(pady=5)

# Start the GUI loop
root.mainloop()