from transformers import GPT2Model, GPT2Tokenizer
import torch

# Load GPT-2 Small model and tokenizer
model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

print("1")

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text and get input IDs
inputs = tokenizer(text, return_tensors="pt")

print("2")

# Get the embeddings from the input layer
with torch.no_grad():
    embeddings = model.get_input_embeddings()(inputs['input_ids'])

print("3")

import numpy as np
import pandas as pd

# Reshape embeddings to 2D (tokens, embedding_dim)
embedding_matrix = embeddings.squeeze(0).cpu().numpy()

print("4")

# Compute the correlation matrix
correlation_matrix = np.corrcoef(embedding_matrix, rowvar=False)

print("5")

# Convert to a DataFrame for easier readability
correlation_df = pd.DataFrame(correlation_matrix, 
                              index=[tokenizer.decode(id) for id in inputs['input_ids'][0]],
                              columns=[tokenizer.decode(id) for id in inputs['input_ids'][0]])

print("6")

# Display the correlation matrix
print(correlation_df)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Token Embedding Correlation Matrix")
plt.show()

print("7")

