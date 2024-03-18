from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.functional as F
import numpy as np

# Load BERT base model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample input text
input_text = "This is a sample input text."

# Tokenize input text
input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True, truncation=True, return_tensors='pt')

# Get the weight matrix to be factorized (example: BERT classifier layer weight matrix)
weight_matrix = bert_model.classifier.weight.detach().numpy()

# Perform Singular Value Decomposition (SVD) to factorize the weight matrix
U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)

# Set the rank for compression (example: keep top 50 singular values)
rank = 50

# Compress the weight matrix using matrix factorization
compressed_weight_matrix = np.dot(U[:, :rank], np.dot(np.diag(S[:rank]), Vt[:rank, :]))

# Update the BERT model with the compressed weight matrix
bert_model.classifier.weight.data = torch.tensor(compressed_weight_matrix)

# Forward pass through the compressed model
with torch.no_grad():
    outputs = bert_model(input_ids)

# Convert logits to probabilities
probabilities = F.softmax(outputs.logits, dim=-1)

# Print the results
print("Compressed Model Output:")
print(torch.argmax(outputs.logits, dim=-1))
print("Probabilities:")
print(probabilities)
