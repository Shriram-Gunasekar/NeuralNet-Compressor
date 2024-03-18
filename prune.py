from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.utils.prune as prune

# Load BERT base model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample input text
input_text = "This is a sample input text."

# Tokenize input text
input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True, truncation=True, return_tensors='pt')

# Pruning parameters
pruning_method = 'l1_unstructured'  # Pruning method
pruning_amount = 0.5  # Percentage of connections to prune

# Define pruning configuration
config = [(torch.nn.Linear, 'weight'), (torch.nn.Linear, 'bias')]

# Apply pruning
prune.global_unstructured(
    parameters_to_prune=config,
    pruning_method=pruning_method,
    amount=pruning_amount,
)

# Forward pass through pruned model
with torch.no_grad():
    outputs = bert_model(input_ids)

# Convert logits to probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Print the results
print("Original Model Output:")
print(torch.argmax(outputs.logits, dim=-1))
print("Probabilities:")
print(probabilities)
