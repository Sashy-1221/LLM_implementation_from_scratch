import torch
import torch.nn as nn

"""
    using this to apply masks to the future tokens. used gpt for proper noes and reference 
"""


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        d_in: input embedding size (e.g., 3)
        d_out: query/key/value size (e.g., 2)
        context_length: max sequence length (e.g., 6)
        dropout: dropout rate (e.g., 0.1)
        qkv_bias: use bias in linear layers? (usually False)

        KEY INNOVATION: Creates causal mask to prevent looking at future tokens
        """
        super().__init__()
        self.d_out = d_out

        # Linear projections: x (d_in) → Q/K/V (d_out)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # CAUSAL MASK: Upper triangle = 1 (block future tokens)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):  # x: (batch, seq_len, d_in)
        b, t, _ = x.shape  # batch, tokens, embedding_dim

        # Compute Q, K, V: (batch, seq_len, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Attention scores: (batch, seq_len, seq_len)
        attn_scores = queries @ keys.transpose(1, 2)  # Q @ K^T

        # APPLY MASK: Set future positions to -∞ (softmax → 0)
        attn_scores.masked_fill_(self.mask.bool()[:t, :t], -torch.inf)

        # Softmax: scores → probabilities (scaled by √d_k)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vectors: weighted sum of values
        context_vec = attn_weights @ values  # (batch, seq_len, d_out)
        return context_vec

# =============================================================================
# FLOW SUMMARY:
# 1. x → Linear(Q,K,V) → Q,K,V (all: batch×seq×d_out)
# 2. Q @ K^T → attn_scores (batch×seq×seq)
# 3. MASK future → -∞ positions
# 4. softmax(√d_k) → attn_weights (probabilities)
# 5. attn_weights @ V → context_vec (batch×seq×d_out)
# =============================================================================

# USAGE:
# ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.1)
# output = ca(inputs)  # (batch, 6, 2)
