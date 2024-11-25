import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber

model_path = 'gpt-neo-125M'
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

# Load the PDF and process
pdf_path = "file.pdf"  # Replace with your actual PDF path
document_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(document_text)

# Print the first few chunks for inspection
if text_chunks:
    print("Extracted text:")
    print(text_chunks[:1])  # Print the first chunk for a sample
else:
    print("No text extracted.")

# Example usage
print(generate_response("What's the capital of France?"))
