from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.quantization import quantize_dynamic

# Load BERT base model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample input text
input_text = "This is a sample input text."

# Tokenize input text
input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True, truncation=True, return_tensors='pt')

# Perform dynamic quantization
quantized_model = quantize_dynamic(bert_model, {torch.nn.Linear}, dtype=torch.qint8)

# Forward pass through quantized model
with torch.no_grad():
    outputs = quantized_model(input_ids)

# Convert logits to probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Print the results
print("Original Model Output:")
print(torch.argmax(outputs.logits, dim=-1))
print("Probabilities:")
print(probabilities)
