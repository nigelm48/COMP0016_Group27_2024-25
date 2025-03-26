# Import full modules for better PyInstaller compatibility
import time
import tkinter as tk
import tkinter.scrolledtext
import tkinter.filedialog
import torch
import transformers
import langchain_chroma
import langchain.prompts
from populate_database import add_documents_to_chroma, clear_database
from embedding import embedding_function
import gc
import os
from PIL import Image, ImageTk

# Create aliases for frequently used classes/functions
sleep = time.sleep
scrolledtext = tkinter.scrolledtext
filedialog = tkinter.filedialog
AutoTokenizer = transformers.AutoTokenizer
AutoModelForCausalLM = transformers.AutoModelForCausalLM
pipeline = transformers.pipeline
Chroma = langchain_chroma.Chroma
PromptTemplate = langchain.prompts.PromptTemplate

# LLM configuration and initialization
#model_path = 'llama3.2-1b'  # Model identifier
#model_path = 'Llama-3.2-3B'  # Model identifier
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "Qwen2.5-1.5B")  
#model_path = 'Qwen2.5-3B'  # Model identifier
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


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

# Initialize text generation pipeline
llm = generator

# Persistent storage directory for vector database
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")

# Configure tokenizer to handle padding correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Conversation tracking container
conversation_history = []

# Font size configuration
default_font_size = 12
current_font_size = default_font_size


# Retrieve embedding model information
def get_embedding_model_name():
    return "multilingual-e5-small"


# Semantic search function with content filtering
def retrieve_similar_documents(query, top_k=5):
    # Retrieve extra documents to ensure sufficient results after filtering
    retrieval_multiplier = 2
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function()
    )

    # Execute semantic similarity search
    results = db.similarity_search(query, k=top_k * retrieval_multiplier)
    db._client._system.stop()
    db=None

    # Apply content filtering if exclusion terms exist
    if do_not_include_items:
        filtered_results = []
        for result in results:
            # Check for excluded content (case-insensitive)
            should_include = True
            for excluded_item in do_not_include_items:
                if excluded_item.lower() in result.page_content.lower():
                    should_include = False
                    break

            if should_include:
                filtered_results.append(result)
                # Early termination if sufficient results collected
                if len(filtered_results) >= top_k:
                    break

        # Warn if insufficient results after filtering
        if len(filtered_results) < top_k:
            print(
                f"Warning: Only {len(filtered_results)} documents remain after filtering, less than requested {top_k}")
            output_box.insert(tk.END,
                              f"Warning: Only {len(filtered_results)} documents remain after filtering, less than requested {top_k}\n",
                              "bot")
            output_box.see(tk.END)
        # Return filtered document content
        return [result.page_content for result in filtered_results[:top_k]]
    else:
        # Return unfiltered document content if no exclusions
        return [result.page_content for result in results[:top_k]]


# Process query and generate AI response
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
            max_new_tokens=200,
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


# Adjust UI text size dynamically
def change_font_size(delta=0):
    global current_font_size
    old_size = current_font_size

    if delta == 0:  # Reset to default value
        current_font_size = default_font_size
    else:
        current_font_size = max(6, min(24, current_font_size + delta))

    # Update main text components
    output_box.config(font=("TkDefaultFont", current_font_size))
    input_box.config(font=("TkDefaultFont", current_font_size))

    # Scale model info labels proportionally
    info_font_size = max(7, int(current_font_size))
    models_label.config(font=("TkDefaultFont", info_font_size, "bold"))
    llm_label.config(font=("TkDefaultFont", info_font_size))
    embedding_label.config(font=("TkDefaultFont", info_font_size))

    # Display warning for excessive font sizes
    if current_font_size > 18 and current_font_size > old_size:
        output_box.insert(tk.END, "Warning! Font size exceeds limit! Controls may be affected!\n", "bot")

    output_box.insert(tk.END, f"Font size set to {current_font_size}\n", "bot")
    output_box.see(tk.END)


# Process user input and generate AI response
def send_query():
    query = input_box.get("1.0", tk.END).strip()
    if query:
        # Display and record user input
        output_box.insert(tk.END, f"You: {query}\n", "user")
        output_box.see(tk.END)
        conversation_history.append({"role": "user", "content": query})

        # Retrieve relevant context and generate response
        similar_docs = retrieve_similar_documents(query)
        context = "\n".join(similar_docs)
        response = generate_response(query, context=context)

        # Display and record AI response
        output_box.insert(tk.END, f"AI: {response}\n\n", "bot")
        output_box.see(tk.END)
        conversation_history.append({"role": "bot", "content": response})

        # Clear input field
        input_box.delete("1.0", tk.END)

# Select and process a single file (Markdown, Word, or PDF)
def browse_file():
    file_selected = filedialog.askopenfilename(
        filetypes=[
            ("Markdown files", "*.md"),
            ("Word documents", "*.docx"),
            ("PDF files", "*.pdf"),
            ("All supported files", "*.md *.docx *.pdf")
        ],
        title="Select a File"
    )

    if file_selected:
        output_box.insert(tk.END, f"Processing file: {file_selected}\n", "bot")
        output_box.see(tk.END)
        root.update_idletasks()
        try:
            # Index the document into the vector database
            add_documents_to_chroma(file_selected)
            output_box.insert(tk.END, "File added to the database successfully.\n", "bot")
        except Exception as e:
            output_box.insert(tk.END, f"Error: {str(e)}\n", "bot")
        output_box.see(tk.END)
    else:
        output_box.insert(tk.END, "No file was selected.\n", "bot")
        output_box.see(tk.END)


# Select and process document folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        output_box.insert(tk.END, f"Processing files in folder: {folder_selected}\n", "bot")
        output_box.see(tk.END)
        root.update_idletasks()
        try:
            # Index documents into vector database
            add_documents_to_chroma(folder_selected)
            output_box.insert(tk.END, "Documents added to the database successfully.\n", "bot")
        except Exception as e:
            output_box.insert(tk.END, f"Error: {str(e)}\n", "bot")
        output_box.see(tk.END)
    else:
        output_box.insert(tk.END, "No folder was selected.\n", "bot")
        output_box.see(tk.END)


# Save conversation history to file
def export_conversation():
    if not conversation_history:
        output_box.insert(tk.END, "No conversation to export.\n", "bot")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", ".txt"), ("All files", ".*")],
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


# Remove all indexed documents from database
def delete_database():
    confirmation = tk.messagebox.askyesno(
        "Confirm Delete",
        "Are you sure you want to delete the entire database?\nThis action cannot be undone."
    )

    if confirmation:
        output_box.insert(tk.END, "Attempting to clear database contents...\n", "bot")
        output_box.see(tk.END)
        root.update_idletasks()  # Force UI update
        
        abs_chroma_path = os.path.abspath(CHROMA_PATH)
        output_box.insert(tk.END, f"Database path: {abs_chroma_path}\n", "bot")
        
        try:
            output_box.insert(tk.END, "Closing database connections...\n", "bot")
            root.update_idletasks()
            
            db = Chroma(
                persist_directory=abs_chroma_path,
                embedding_function=embedding_function()
            )
            
            db._client._system.stop()
            db = None
            gc.collect()
            time.sleep(0.5)  
            
        except Exception as e:
            output_box.insert(tk.END, f"Warning during connection cleanup: {str(e)}\n", "bot")
        
        # Now delete contents 
        success = False
        try:
            if os.path.exists(abs_chroma_path):
                for item in os.listdir(abs_chroma_path):
                    item_path = os.path.join(abs_chroma_path, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                            output_box.insert(tk.END, f"Deleted file: {item}\n", "bot")
                        elif os.path.isdir(item_path):
                            import shutil
                            shutil.rmtree(item_path)
                            output_box.insert(tk.END, f"Deleted directory: {item}\n", "bot")
                    except Exception as e:
                        output_box.insert(tk.END, f"Failed to delete {item}: {str(e)}\n", "bot")
                
                try:
                    with open(os.path.join(abs_chroma_path, ".empty"), "w") as f:
                        f.write("# This file ensures the chroma directory exists and is writable")
                    success = True
                except Exception as e:
                    output_box.insert(tk.END, f"Warning: Could not write test file: {str(e)}\n", "bot")
            else:
                os.makedirs(abs_chroma_path, exist_ok=True)
                success = True
                output_box.insert(tk.END, "Created empty database directory.\n", "bot")
        except Exception as e:
            output_box.insert(tk.END, f"Error clearing database contents: {str(e)}\n", "bot")
        
        if success:
            output_box.insert(tk.END, "Database contents cleared successfully.\n", "bot")
        else:
            output_box.insert(tk.END, "WARNING: Database may not be fully cleared.\n", "bot")
            
        output_box.insert(tk.END, "Please restart the application for changes to take effect fully.\n", "bot")
        output_box.see(tk.END)


# Clean up resources when application closes
def on_closing():
    try:
        # Close database connections
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function()
        )

        # Terminate client connections
                    
        db._client._system.stop()
        db = None

        # Release embedding model resources
        try:
            emb_func = embedding_function()
            if hasattr(emb_func, 'model'):
                emb_func.model = None
                del emb_func.model
        except:
            pass

        # Release LLM resources
        global model, tokenizer, generator
        model = None
        tokenizer = None
        generator = None

        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.mps.is_available():
            gc.collect()
        sleep(0.2)  # Allow time for memory release

    except Exception as e:
        print(f"Error during cleanup: {e}")

    # Close application
    root.destroy()


def show_help():
    # Create a new toplevel window for help
    help_window = tk.Toplevel(root)
    help_window.title("AI RAG Assistant Help")
    help_window.geometry("700x500")
    help_window.minsize(700, 700)
    
    # Create a scrollable text area for the help content
    help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, 
                                         font=("TkDefaultFont", current_font_size+2))
    help_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    # Add the help content
    help_content = """# AI RAG Assistant Help

## Overview
This application uses Retrieval-Augmented Generation (RAG) to provide accurate answers from your documents.

## Basic Features
- **Load Folder**: Import multiple documents from a directory
- **Load File**: Import a single document (PDF, Word, Markdown)
- **Delete Database**: Remove all indexed documents
- **Export Chat**: Save the current conversation to a text file
- **Send**: Process your query and get an AI response
- **Font Size**: Adjust text size for better readability
- **Search**: Find text in the chat history

## Do-not-include Items
- **Add**: Enter keywords to exclude from search results
- **Remove**: Select and remove items from the exclusion list
- **Filter**: Search for specific items in the exclusion list
- **Sort**: Arrange items in alphabetical order

## Prompt Recommendations
- **No prompting is specifically required, as they would downgrade the overall performance of the model.**
- **However, please include as many keywords as possible in your query to improve search relevance.**
- **The model will try to find the most relevant information based on the query and the do-not-include-items.**

## Example Usage
- **Question**: What happened in Ghana?  //this should only be asked when the number of ducuments is small
  **Answer**: In Ghana, people struggle with speech articulation due to various conditions, making it difficult for them to be understood by others. The use of Automatic Speech Recognition (ASR) software, such as Google's "Talk-to-Text" feature, can help improve communication by enabling listeners to better understand the speaker. This technology has the potential to bridge the communication gap and enhance the quality of life for those affected by speech and language difficulties. Additionally, it can facilitate access to information and services that may have been previously inaccessible to them. The study highlights the importance of considering the local context and infrastructure when implementing such technologies, particularly in low-income countries like Ghana where internet connectivity and data costs can pose significant challenges. By understanding the needs and experiences of users, researchers and developers can design more effective and accessible solutions that address the unique challenges faced by individuals in these communities.

- **Question**: What will researchers do if the participant doesn't have a phone?  
  **Answer**: Researchers will provide the smartphone to them. They will also provide compensation to those who already have a suitable smartphone. The compensation will be approximately $120. The researchers also ensured that the participation is voluntary, participants can withdraw from the experiment at any time, they can keep the provided smartphone or compensation and their participation will not impact their future speech and language therapy (SLT) services. Additionally, the researchers will help them use the app if they need assistance and will support them in getting their voice heard by lobbying for the changes they requested. The participants will have an onboarding session with researchers at their office or at a location of their choice, where they will discuss their communication difficulties and how they currently use their mobile devices. This session will last for about 30 minutes. During this session, a researcher will explain the purpose and procedures of this study and answer any questions they may have. After the session is completed, researchers and participants sign an informed consent form.
"""
    
    help_text.insert(tk.END, help_content)
    help_text.config(state=tk.DISABLED)  # Make read-only
    
    # Add a close button
    close_btn = tk.Button(help_window, text="Close", height=BUTTON_HEIGHT,
                         command=help_window.destroy)
    close_btn.pack(pady=10)


def search_text():
    """Search for text"""
    # Clear previous highlights
    start_pos = "1.0"
    while True:
        tag_range = output_box.tag_nextrange("search", start_pos)
        if not tag_range:
            break
        output_box.tag_remove("search", tag_range[0], tag_range[1])
        start_pos = tag_range[1]
    # Perform new search
    search_term = search_entry.get().strip()
    if not search_term:
        return
    output_box.tag_configure("search", background="orange", foreground="black")
    start_pos = "1.0"
    matches_counter = 0
    while True:
        start_pos = output_box.search(search_term, start_pos, tk.END, nocase=1)
        if not start_pos:
            break
        end_pos = f"{start_pos}+{len(search_term)}c"
        # Apply highlight tag
        output_box.tag_add("search", start_pos, end_pos)
        matches_counter += 1
        start_pos = end_pos
    
    search_status.config(text=f"Total: {matches_counter} matches")
    if matches_counter > 0:
        output_box.see("search.first")
        output_box.tag_add(tk.SEL, "search.first", "search.first lineend")

def search_navigation(direction="next"):
    """Navigate to the next or previous search match based on the direction"""
    # Get current insert position
    current_pos = output_box.index(tk.INSERT)
    
    try:
        if direction == "next":
            match = output_box.tag_nextrange("search", current_pos)
            if not match:
                # Wrap around to the first match if no more matches are found
                match = output_box.tag_nextrange("search", "1.0")
        elif direction == "prev":
            match = output_box.tag_prevrange("search", current_pos)
            if not match:
                # Wrap around to the last match if no more matches are found
                match = output_box.tag_prevrange("search", tk.END)
        else:
            raise ValueError("Invalid direction. Use 'next' or 'prev'.")
        
        if match:
            output_box.see(match[0])
            output_box.mark_set(tk.INSERT, match[0])
    except Exception as e:
        pass

def search_next():
    search_navigation(direction="next")

def search_prev():
    search_navigation(direction="prev")

def clear_search():
    """Clear search"""
    start_pos = "1.0"
    while True:
        tag_range = output_box.tag_nextrange("search", start_pos)
        if not tag_range:
            break
        output_box.tag_remove("search", tag_range[0], tag_range[1])
        start_pos = tag_range[1]    
    search_entry.delete(0, tk.END)
    search_status.config(text="")

# Initialize main application window
root = tk.Tk()
root.title("AI RAG Assistant")

# Configure responsive window sizing
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{int(screen_width * 0.75)}x{int(screen_height * 0.75)}")
root.minsize(900, 700)

# Configure layout responsiveness
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

# Create a logo frame at the top right
logo_frame = tk.Frame(root)
logo_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ne")

# Load and display the logo
try:
    # Path to your logo image - adjust the filename to match your logo
    logo_path = os.path.join(current_dir, "UCL-logo.png")
    
    # Open and resize the image
    logo_image = Image.open(logo_path)
    # Resize while maintaining aspect ratio
    logo_width = 120  # Set your desired width
    aspect_ratio = logo_image.width / logo_image.height
    logo_height = int(logo_width / aspect_ratio)
    logo_image = logo_image.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
    
    # Convert to PhotoImage for tkinter
    logo_tk = ImageTk.PhotoImage(logo_image)
    
    # Create label to display logo
    logo_label = tk.Label(logo_frame, image=logo_tk)
    logo_label.image = logo_tk  # Keep reference to prevent garbage collection
    logo_label.pack()
    
except Exception as e:
    print(f"Error loading logo: {e}")

# Font size control panel - now below the logo
controls_frame = tk.Frame(root)
controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(logo_height + 15, 5), sticky="ne")

font_label = tk.Label(controls_frame, text="Font Size:")
font_label.pack(side=tk.LEFT, padx=5)

# Define a standard button height for the application
BUTTON_HEIGHT = 2

# Modify the font size control buttons
decrease_font_btn = tk.Button(controls_frame, text="-", width=2, height=BUTTON_HEIGHT, command=lambda: change_font_size(-1))
decrease_font_btn.pack(side=tk.LEFT, padx=2)

reset_font_btn = tk.Button(controls_frame, text="Reset", height=BUTTON_HEIGHT, command=lambda: change_font_size(0))
reset_font_btn.pack(side=tk.LEFT, padx=2)

increase_font_btn = tk.Button(controls_frame, text="+", width=2, height=BUTTON_HEIGHT, command=lambda: change_font_size(1))
increase_font_btn.pack(side=tk.LEFT, padx=2)

# Model information display
llm_model_name = model_path
embedding_model_name = get_embedding_model_name()

model_info_frame = tk.Frame(root)
model_info_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nw")

models_label = tk.Label(model_info_frame, text="Models:", font=("TkDefaultFont", current_font_size+1, "bold"))
models_label.pack(side=tk.LEFT, padx=5)

llm_label = tk.Label(model_info_frame, text=f"LLM: {llm_model_name}", font=("TkDefaultFont", current_font_size+1))
llm_label.pack(side=tk.LEFT, padx=5)

embedding_label = tk.Label(model_info_frame, text=f"Embedding: {embedding_model_name}", font=("TkDefaultFont", current_font_size+1))
embedding_label.pack(side=tk.LEFT, padx=5)

# Create search frame 
search_frame = tk.Frame(root)
search_frame.grid(row=0, column=0, padx=10, pady=(50, 5), sticky="nw")  

# Search label
search_label = tk.Label(search_frame, text="Search:")
search_label.pack(side=tk.LEFT, padx=(0, 5))

# Search entry
search_entry = tk.Entry(search_frame, width=20)
search_entry.pack(side=tk.LEFT, padx=5)
# Search buttons
search_btn = tk.Button(search_frame, text="Find", height=1, command=search_text)
search_btn.pack(side=tk.LEFT, padx=2)

prev_btn = tk.Button(search_frame, text="<-", height=1, width=2, command=search_prev)
prev_btn.pack(side=tk.LEFT, padx=2)

next_btn = tk.Button(search_frame, text="->", height=1, width=2, command=search_next)
next_btn.pack(side=tk.LEFT, padx=2)

clear_btn = tk.Button(search_frame, text="Clear", height=1, command=clear_search)
clear_btn.pack(side=tk.LEFT, padx=2)

search_status = tk.Label(search_frame, text="", width=12)
search_status.pack(side=tk.LEFT, padx=5)

# Help frame now positioned BELOW the search frame
help_frame = tk.Frame(root)
help_frame.grid(row=0, column=0, padx=10, pady=(110, 5), sticky="nw")  # More padding to push it down

help_btn = tk.Button(help_frame, text="Help", height=BUTTON_HEIGHT, command=show_help)
help_btn.pack(side=tk.LEFT, padx=5)

# Chat display area - REMOVE the duplicate search frame that was here previously
output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=60,
                                       font=("TkDefaultFont", current_font_size))
output_box.tag_config("user", foreground="blue")
output_box.tag_config("bot", foreground="green")
output_box.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# User input area
input_box = tk.Text(root, height=3, font=("TkDefaultFont", current_font_size))
input_box.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

# Action button container
button_bar = tk.Frame(root)
button_bar.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

# Action buttons
browse_button = tk.Button(button_bar, text="Load Folder", height=BUTTON_HEIGHT, command=browse_folder)
browse_button.pack(side=tk.LEFT, padx=5, expand=True)

# Add "Load File" button
load_file_button = tk.Button(button_bar, text="Load File", height=BUTTON_HEIGHT, command=browse_file)
load_file_button.pack(side=tk.LEFT, padx=5, expand=True)

delete_db_button = tk.Button(button_bar, text="Delete Database", height=BUTTON_HEIGHT, command=delete_database, fg="red")
delete_db_button.pack(side=tk.LEFT, padx=5, expand=True)

export_button = tk.Button(button_bar, text="Export Chat", height=BUTTON_HEIGHT, command=export_conversation)
export_button.pack(side=tk.LEFT, padx=5, expand=True)

send_button = tk.Button(button_bar, text="Send", height=BUTTON_HEIGHT, command=send_query)
send_button.pack(side=tk.LEFT, padx=5, expand=True)

# Layout weight configuration
root.columnconfigure(0, weight=3)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=5)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)

# Content exclusion list management
do_not_include_items = []


def update_do_not_include_listbox(items):
    do_not_include_listbox.delete(0, tk.END)
    for item in items:
        do_not_include_listbox.insert(tk.END, item)


def add_do_not_include_item():
    item = do_not_include_entry.get().strip()
    if item:
        do_not_include_items.append(item)
        update_do_not_include_listbox(do_not_include_items)
        do_not_include_entry.delete(0, tk.END)


def filter_do_not_include_items():
    filter_text = filter_entry.get().strip()
    if filter_text:
        filtered = [item for item in do_not_include_items if filter_text.lower() in item.lower()]
        update_do_not_include_listbox(filtered)
    else:
        update_do_not_include_listbox(do_not_include_items)


def sort_do_not_include_items():
    do_not_include_items.sort()
    update_do_not_include_listbox(do_not_include_items)


def remove_do_not_include_item():
    selected_indices = do_not_include_listbox.curselection()
    if not selected_indices:
        output_box.insert(tk.END, "Please select an item to remove.\n", "bot")
        output_box.see(tk.END)
        return

    # Process selected items in reverse order to maintain correct indexing
    for index in sorted(selected_indices, reverse=True):
        if 0 <= index < len(do_not_include_items):
            removed_item = do_not_include_items.pop(index)
            output_box.insert(tk.END, f"Removed item: {removed_item}\n", "bot")

    # Refresh display
    update_do_not_include_listbox(do_not_include_items)
    output_box.see(tk.END)


# Content exclusion interface
do_not_include_frame = tk.Frame(root)
do_not_include_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

do_not_include_frame.columnconfigure(1, weight=1)  # Make entry expand
do_not_include_frame.rowconfigure(1, weight=1)     # Make listbox expand vertically

# Input controls for exclusion items
do_not_include_label = tk.Label(do_not_include_frame, text="Do Not Include Items:")
do_not_include_label.grid(row=0, column=0, sticky="w")

do_not_include_entry = tk.Entry(do_not_include_frame)
do_not_include_entry.grid(row=0, column=1, padx=5, sticky="ew")  # Make entry expand horizontally

add_item_btn = tk.Button(do_not_include_frame, text="Add", height=BUTTON_HEIGHT, command=add_do_not_include_item)
add_item_btn.grid(row=0, column=2, padx=5)

remove_item_btn = tk.Button(do_not_include_frame, text="Remove", height=BUTTON_HEIGHT, command=remove_do_not_include_item, fg="red")
remove_item_btn.grid(row=0, column=3, padx=5)

# Exclusion item display - make it responsive
do_not_include_listbox = tk.Listbox(do_not_include_frame, height=5, selectmode=tk.EXTENDED)
do_not_include_listbox.grid(row=1, column=0, columnspan=4, pady=5, sticky="nsew")

# Filtering and sorting controls
filter_controls_frame = tk.Frame(do_not_include_frame)
filter_controls_frame.grid(row=2, column=0, columnspan=4, pady=5, sticky="ew")  # Make it span all columns

filter_controls_frame.columnconfigure(1, weight=1)  # Make filter entry expand

filter_label = tk.Label(filter_controls_frame, text="Filter:")
filter_label.grid(row=0, column=0, sticky="w", padx=(0, 5))

filter_entry = tk.Entry(filter_controls_frame)
filter_entry.grid(row=0, column=1, sticky="ew", padx=5)  # Make it expand horizontally

filter_btn = tk.Button(filter_controls_frame, text="Apply Filter", height=BUTTON_HEIGHT, command=filter_do_not_include_items)
filter_btn.grid(row=0, column=2, padx=5)

sort_btn = tk.Button(filter_controls_frame, text="Sort Items", height=BUTTON_HEIGHT, command=sort_do_not_include_items)
sort_btn.grid(row=0, column=3, padx=5)

# Set up window close handler
root.protocol("WM_DELETE_WINDOW", on_closing)

# Launch application
root.mainloop()
