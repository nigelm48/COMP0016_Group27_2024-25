import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber

model_path = 'gpt-neo-125m'  # Path where the files are saved
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Extract text from a PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  # Extracts text for each page
    return text

# Load the PDF and extract text
pdf_path = "file.pdf"  # Replace with your actual PDF path
document_text = extract_text_from_pdf(pdf_path)

# Split text into chunks (for better vectorization)
def chunk_text(text, chunk_size=500):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

text_chunks = chunk_text(document_text)

# Print the first few chunks for inspection
if text_chunks:
    print("Extracted text:")
    print(text_chunks[:1])  # Print the first chunk for a sample
else:
    print("No text extracted.")

# Function to generate responses using the model
def generate_response(input_text):
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set eos_token as pad_token

    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Ensure the model is on the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Generate output using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            pad_token_id=tokenizer.pad_token_id,  # Now it uses pad_token_id
            max_length=512,  # Adjust based on your needs
            num_beams=5,  # Beam search for better quality
            no_repeat_ngram_size=2,  # Prevent repeating text
            early_stopping=True
        )

    # Decode the generated output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)