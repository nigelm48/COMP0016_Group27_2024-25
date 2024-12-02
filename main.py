import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber
from docx import Document
import markdown
import re

model_path = 'llama3.2-1b'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Extract text from a PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  # Extracts text for each page
    return text

# Extract text from a Word document using python-docx
def extract_text_from_docx(docx_path):
    document = Document(docx_path)
    return "\n".join([para.text for para in document.paragraphs if para.text.strip()])

# Extract text from a Markdown file using markdown and remove HTML tags
def extract_text_from_markdown(md_path):
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    # Convert Markdown to HTML, then clean out HTML tags to get plain text
    html_content = markdown.markdown(md_content)
    # Remove HTML tags to produce clean text
    plain_text = re.sub(r'<[^>]+>', '', html_content)
    return plain_text

# Chunk text into manageable sizes
def chunk_text(text, chunk_size=256):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Generate response
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,  # Specify the max number of new tokens to generate
            temperature = 0.1,  # Set temperature to 0 for deterministic output
            top_p=0.9,
            num_beams=3,
            no_repeat_ngram_size=2,
        )

    generated_text =  tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Trim the response to the last full stop
    if '.' in generated_text:
        last_full_stop_index = generated_text.rfind('.')  # Find the last full stop
        trimmed_response = generated_text[:last_full_stop_index + 1]  # Include the full stop
    else:
        trimmed_response = generated_text  # Return as is if no full stop is found

    return trimmed_response


def process_documents(folder_path):
    results = []
    try:
        files = os.listdir(folder_path)
        if files:
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                try:
                    if file_name.endswith('.pdf'):
                        print(f"Processing PDF file: {file_name}")
                        document_text = extract_text_from_pdf(file_path)
                    elif file_name.endswith('.docx'):
                        print(f"Processing Word document: {file_name}")
                        document_text = extract_text_from_docx(file_path)
                    elif file_name.endswith('.md'):
                        print(f"Processing Markdown file: {file_name}")
                        document_text = extract_text_from_markdown(file_path)
                    else:
                        results.append((file_name, "Unsupported file type"))
                        continue

                    # Chunk text for easier processing
                    text_chunks = chunk_text(document_text)

                    # Print text chunks to the console
                    print(f"\nExtracted text from {file_name}:")
                    for chunk in text_chunks:
                        print(chunk)

                    # Append success to results
                    results.append((file_name, text_chunks))
                except Exception as e:
                    results.append((file_name, f"Error: {str(e)}"))
        else:
            results.append(("No files found", "Error: Folder is empty"))
    except Exception as e:
        results.append(("Error reading folder", f"Error: {str(e)}"))

    return results


# Example usage
print(generate_response("What's the capital of France?"))
