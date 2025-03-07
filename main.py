from time import sleep
import tkinter as tk
from tkinter import scrolledtext, filedialog
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from populate_database import add_documents_to_chroma
from embedding import embedding_function

# Initialize model and tokenizer
model_path = 'Llama-3.2-3B-Instruct'  # Replace with your model's path
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  
).to(device)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=3000,
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

# Add message storage
conversation_history = []

default_font_size = 10  # Starting font size
current_font_size = default_font_size

# Get embedding model name from the embedding function
def get_embedding_model_name():
    return "multilingual-e5-large"

# Function to retrieve similar documents
def retrieve_similar_documents(query, top_k=5):
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
    if (last_period_index != -1):
        answer = answer[:last_period_index + 1]
    answer = answer.replace("\n", "").replace("\r", "").replace("\t", "")

    return answer

# Function to change font size
def change_font_size(delta=0):
    global current_font_size
    old_size = current_font_size
    
    if delta == 0:  # Reset to default
        current_font_size = default_font_size
    else:
        current_font_size = max(6, min(24, current_font_size + delta))
    
    # Apply font size to text areas
    output_box.config(font=("TkDefaultFont", current_font_size))
    input_box.config(font=("TkDefaultFont", current_font_size))
    
    # Apply smaller font size to model info labels (70% of main font size)
    info_font_size = max(7, int(current_font_size * 0.7))
    models_label.config(font=("TkDefaultFont", info_font_size, "bold"))
    llm_label.config(font=("TkDefaultFont", info_font_size))
    embedding_label.config(font=("TkDefaultFont", info_font_size))
    
    # Show warning if font size is too large
    if current_font_size > 18 and current_font_size > old_size:
        output_box.insert(tk.END, "Warning! Font size exceeds limit! Controls may be affected!\n", "bot")

    output_box.insert(tk.END, f"Font size set to {current_font_size}\n", "bot")
    output_box.see(tk.END)

# Function to send query from the GUI
def send_query():
    query = input_box.get("1.0", tk.END).strip()
    if query:
        # Display user message
        output_box.insert(tk.END, f"You: {query}\n", "user")
        output_box.see(tk.END)

        # Store user message
        conversation_history.append({"role": "user", "content": query})

        # Get response
        similar_docs = retrieve_similar_documents(query)
        context = "\n".join(similar_docs)
        response = generate_response(query, context=context)

        # Display bot message
        output_box.insert(tk.END, f"AI: {response}\n\n", "bot")
        output_box.see(tk.END)

        # Store bot message
        conversation_history.append({"role": "bot", "content": response})

        input_box.delete("1.0", tk.END)

# Function to browse and process a folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        output_box.insert(tk.END, f"Processing files in folder: {folder_selected}\n", "bot")
        output_box.see(tk.END)
        root.update_idletasks()
        try:
            # Add documents to the Chroma vector store using populate_database.py
            add_documents_to_chroma(folder_selected)
            output_box.insert(tk.END, "Documents added to the database successfully.\n", "bot")
        except Exception as e:
            output_box.insert(tk.END, f"Error: {str(e)}\n", "bot")
        output_box.see(tk.END)
    else:
        output_box.insert(tk.END, "No folder was selected.\n", "bot")
        output_box.see(tk.END)

def export_conversation():
    if not conversation_history:
        output_box.insert(tk.END, "No conversation to export.\n", "bot")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="Export Conversation"
    )

    if file_path:
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for message in conversation_history:
                    role = "You" if message["role"] == "user" else "AI"
                    file.write(f"{role}: {message['content']}\n\n")

            output_box.insert(tk.END, f"Conversation exported to {file_path}\n", "bot")
        except Exception as e:
            output_box.insert(tk.END, f"Error exporting conversation: {str(e)}\n", "bot")

        output_box.see(tk.END)


# Create the main window
root = tk.Tk()
root.title("AI RAG Assistant")

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set initial window size (responsive)
root.geometry(f"{int(screen_width * 0.5)}x{int(screen_height * 0.5)}")  # 50% of screen size
root.minsize(600, 400)  # Minimum size to prevent extreme shrinkage

# Configure rows and columns for responsiveness
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)  

# Add a controls frame at the top
controls_frame = tk.Frame(root)
controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ne")

# Font size buttons in the controls frame
font_label = tk.Label(controls_frame, text="Font Size:")
font_label.pack(side=tk.LEFT, padx=5)

decrease_font_btn = tk.Button(controls_frame, text="-", width=2, command=lambda: change_font_size(-1))
decrease_font_btn.pack(side=tk.LEFT, padx=2)

reset_font_btn = tk.Button(controls_frame, text="Reset", command=lambda: change_font_size(0))
reset_font_btn.pack(side=tk.LEFT, padx=2)

increase_font_btn = tk.Button(controls_frame, text="+", width=2, command=lambda: change_font_size(1))
increase_font_btn.pack(side=tk.LEFT, padx=2)

# After initializing the models and before creating the main window
# Add these variables to store model information
llm_model_name = model_path  # This already exists as 'llama3.2-1b'
embedding_model_name = get_embedding_model_name()

# After creating the controls_frame for font size buttons
# Add a new frame for model information display
model_info_frame = tk.Frame(root)
model_info_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nw")

# Add labels to display model information
models_label = tk.Label(model_info_frame, text="Models:", font=("TkDefaultFont", 9, "bold"))
models_label.pack(side=tk.LEFT, padx=5)

llm_label = tk.Label(model_info_frame, text=f"LLM: {llm_model_name}", font=("TkDefaultFont", 9))
llm_label.pack(side=tk.LEFT, padx=5)

embedding_label = tk.Label(model_info_frame, text=f"Embedding: {embedding_model_name}", font=("TkDefaultFont", 9))
embedding_label.pack(side=tk.LEFT, padx=5)

# Output box (Scrollable)
output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=60, font=("TkDefaultFont", current_font_size))
output_box.tag_config("user", foreground="blue")
output_box.tag_config("bot", foreground="green")
output_box.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Input box
input_box = tk.Text(root, height=3, font=("TkDefaultFont", current_font_size))
input_box.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

button_bar = tk.Frame(root)
button_bar.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

browse_button = tk.Button(button_bar, text="Load Documents", command=browse_folder)
browse_button.pack(side=tk.LEFT, padx=5, expand=True)

export_button = tk.Button(button_bar, text="Export Chat", command=export_conversation)
export_button.pack(side=tk.LEFT, padx=5, expand=True)

send_button = tk.Button(button_bar, text="Send", command=send_query)
send_button.pack(side=tk.LEFT, padx=5, expand=True)

# Ensure all columns expand 
root.columnconfigure(0, weight=3)  
root.columnconfigure(1, weight=1)  
root.rowconfigure(0, weight=1)  
root.rowconfigure(1, weight=5)  
root.rowconfigure(2, weight=1)  
root.rowconfigure(3, weight=1)  

# Start the GUI loop
root.mainloop()