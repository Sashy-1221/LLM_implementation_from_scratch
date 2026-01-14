import torch
import torch.nn as nn

# from causal_attention import CausalAttention

# class MultiHeadAttentionWrapper(nn.Module): ---> class to extend causal_attention across multiple heads (simple extension)
#     def __init__(self, num_heads, d_in, d_out,
#                  context_length, dropout, qkv_bias=False):
#         super().__init__()
#         assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
#
#         self.d_out = d_out
#         self.num_heads = num_heads
#         self.heads = nn.ModuleList([
#             CausalAttention(d_in, d_out // num_heads,
#                             context_length, dropout, qkv_bias)
#             for _ in range(num_heads)
#         ])
#
#     def forward(self, x):
#         head_outputs = [head(x) for head in self.heads]
#         context_vec = torch.cat(head_outputs, dim=-1)
#         return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Ensure output dimension is divisible by number of heads
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Linear projections for Query, Key, Value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection after concatenating heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout on attention weights
        self.dropout = nn.Dropout(dropout)

        # Causal mask (prevents attending to future tokens)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):
        # x shape: (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        # Project inputs to queries, keys, values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to (batch, num_heads, num_tokens, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Move num_heads dimension forward
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask (block future positions)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize scores to probabilities
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        # Dropout on attention probabilities
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Concatenate heads back
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        # Final linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec
