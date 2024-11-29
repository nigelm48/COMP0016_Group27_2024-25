import tkinter as tk
from tkinter import scrolledtext, filedialog
from main import generate_response, extract_text_from_pdf, extract_text_from_docx, extract_text_from_markdown, chunk_text, process_documents
import os

def send_query():
    query = input_box.get("1.0", tk.END).strip()
    if query:
        output_box.insert(tk.END, f"You: {query}\n", "user")
        response = generate_response(query)  # Call backend function
        output_box.insert(tk.END, f"AI: {response}\n\n", "bot")
        input_box.delete("1.0", tk.END)


def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        output_box.insert(tk.END, f"Processing files in folder: {folder_selected}\n", "bot")

        # Call process_documents to extract text from suitable files
        results = process_documents(folder_selected)

        # Command-line output
        print(f"Files processed in folder: {folder_selected}")

        # Track if any suitable files were processed
        suitable_files_found = False

        # Check for errors and unsupported files
        errors = [result for result in results if "Error" in result[1]]
        unsupported_files = [result for result in results if "Unsupported file type" in result[1]]

        for file_name, content in results:
            if isinstance(content, list):  # Extracted text chunks
                suitable_files_found = True
                print(f"\nExtracted text from {file_name}:")
                for chunk in content:
                    print(chunk)
            elif "Error" in content or "Unsupported" in content:  # Handle errors and unsupported files
                print(f"\n{file_name}: {content}")

        # GUI output summary
        if errors:
            output_box.insert(tk.END, f"Some files could not be processed due to errors:\n", "bot")
            for file_name, error in errors:
                output_box.insert(tk.END, f" - {file_name}: {error}\n", "bot")

        if unsupported_files:
            output_box.insert(tk.END, f"The following files were skipped as unsupported:\n", "bot")
            for file_name, _ in unsupported_files:
                output_box.insert(tk.END, f" - {file_name}\n", "bot")

        # If no suitable files were found, show this message in both GUI and command line
        if not suitable_files_found:
            output_box.insert(tk.END, "No suitable files (PDF, DOCX, or MD) found in the folder.\n", "bot")
            print("No suitable files (PDF, DOCX, or MD) found in the folder.")

        else:
            output_box.insert(tk.END, "Text extraction and analysis completed successfully!\n", "bot")


# Create main window
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
browse_button = tk.Button(root, text="Browse Folder", command=browse_folder)
browse_button.pack(pady=5)

root.mainloop()
