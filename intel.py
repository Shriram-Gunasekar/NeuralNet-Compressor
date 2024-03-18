from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader
from neural_compressor import Config, Encoder, PostTrainingCompressor

# Load BERT base model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample input text
input_text = "This is a sample input text."

# Tokenize input text
input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True, truncation=True, return_tensors='pt')

# Define configuration for Intel Neural Compressor
config = Config({
    'device': 'cpu',  # Specify the device (e.g., 'cpu', 'gpu')
    'compression_algorithm': 'HuffmanCoding',  # Compression algorithm (e.g., 'HuffmanCoding', 'Thresholding')
    'bitwidth': 8,  # Quantization bitwidth
    'quantize_weights': True,  # Quantize weights
    'quantize_activations': True,  # Quantize activations
    'prune_ratio': 0.5,  # Pruning ratio
})

# Create an encoder for the BERT model
encoder = Encoder(config=config, model=bert_model, tokenizer=bert_tokenizer)

# Create a data loader (not required for compression, just for demonstration)
data_loader = DataLoader([input_ids])

# Initialize the post-training compressor
compressor = PostTrainingCompressor(encoder=encoder, config=config)

# Compress the model
compressed_model = compressor.compress_model()

# Forward pass through the compressed model
with torch.no_grad():
    outputs = compressed_model(input_ids)

# Convert logits to probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Print the results
print("Compressed Model Output:")
print(torch.argmax(outputs.logits, dim=-1))
print("Probabilities:")
print(probabilities)
