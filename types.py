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

# Define pruning parameters
pruning_amount = 0.5  # Percentage of connections to prune

# Define pruning method: magnitude-based pruning
def prune_model(module, amount):
    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

# Apply magnitude-based pruning to BERT model
bert_model.apply(lambda module: prune_model(module, pruning_amount))

# Forward pass through pruned model
with torch.no_grad():
    outputs = bert_model(input_ids)

# Convert logits to probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Print the results
print("Pruned Model Output:")
print(torch.argmax(outputs.logits, dim=-1))
print("Probabilities:")
print(probabilities)
