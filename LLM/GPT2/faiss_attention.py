import faiss
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class FAISSAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, k_neighbors=10):
        super().__init__(config, is_cross_attention, layer_idx)
        self.k_neighbors = k_neighbors

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.head_dim)  # Exact search; replace with an approximate method if needed

    def add_keys_to_faiss(self, key):
        # Convert keys to numpy format for FAISS
        key_np = key.detach().cpu().numpy().astype(np.float32)
        
        #if self.index.ntotal == 0:
        self.index.add(key_np.reshape(-1, self.head_dim))

    def faiss_based_attention(self, query, value):
        print("attention entered")
        query_np = query.detach().cpu().numpy().astype(np.float32).reshape(-1, self.head_dim)
        print("Query shape:",query_np.shape)
        distances, indices = self.index.search(query_np, self.k_neighbors)
        print("Search done")
        indices = torch.tensor(indices, device=value.device)
        gathered_values = torch.gather(value, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, self.head_dim))
        attn_output = gathered_values.mean(dim=1) 
        
        return attn_output, distances

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        print("Started forward call")
        
        # Generate query, key, and value vectors
        if encoder_hidden_states is not None:
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        print("Generated query, key, and value vectors")

        # Split into heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        print("Split the heads")

        # Add keys to the FAISS index
        self.add_keys_to_faiss(key)

        print("Added the keys to Faiss")
        print(self.index.ntotal)

        # FAISS-based similarity search for queries
        attn_output, attn_weights = self.faiss_based_attention(query, value)

        print("Search done")

        # Merge heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        print("Waffle")

        outputs = (attn_output, None)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
