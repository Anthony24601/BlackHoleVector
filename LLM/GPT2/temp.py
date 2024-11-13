import torch
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import GPT2Model, GPT2Config
from faiss_attention import * 

class GPT2FaissBlock(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        self.attn = FAISSAttention(config)

class GPT2FaissModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.h)):
            self.h[i] = GPT2FaissBlock(config)

model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name)
config = GPT2Config.from_pretrained("gpt2")

custom_model = GPT2FaissModel(config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model(**inputs)
print(outputs)

# Forward pass through the custom model
outputs = custom_model(**inputs)
print(outputs)