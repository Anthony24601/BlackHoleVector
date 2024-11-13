import faiss
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class RollingFAISSAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, k_neighbors=10, max_index_size=100000):
        super().__init__(config, is_cross_attention, layer_idx)
        self.k_neighbors = k_neighbors
        self.max_index_size = max_index_size  # Maximum number of vectors to store in FAISS index

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.head_dim)  # Exact search; replace with approximate if needed

        # Keep a deque for rolling storage of key vectors
        self.key_queue = deque(maxlen=max_index_size)
    
    def add_keys_to_faiss(self, key):
        # Convert keys to numpy format for FAISS
        key_np = key.detach().cpu().numpy().astype(np.float32)

        # If we exceed the max capacity, we reset the index and re-add the most recent keys
        if len(self.key_queue) + key_np.shape[0] > self.max_index_size:
            # Reset index and re-add from the queue if capacity exceeded
            self.index.reset()
            all_keys = np.vstack(list(self.key_queue))  # Combine all keys in the queue
            self.index.add(all_keys)
        
        # Add new keys to the FAISS index and update the queue
        self.index.add(key_np)
        self.key_queue.extend(key_np)  # Append new keys to the queue

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
        
        # Generate query, key, and value vectors
        if encoder_hidden_states is not None:
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # Split into heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Add keys to the FAISS index
        self.add_keys_to_faiss(key)

        # FAISS-based similarity search for queries
        attn_output, attn_weights = self.faiss_based_attention(query, value)

        # Merge heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, None)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
