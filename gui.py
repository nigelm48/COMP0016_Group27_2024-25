import tkinter as tk
from tkinter import scrolledtext
from main import generate_response  # Replace with actual backend function.

def send_query():
    query = input_box.get("1.0", tk.END).strip()
    if query:
        output_box.insert(tk.END, f"You: {query}\n", "user")
        response = generate_response(query)  # Call backend function
        output_box.insert(tk.END, f"AI: {response}\n\n", "bot")
        input_box.delete("1.0", tk.END)

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

root.mainloop()
