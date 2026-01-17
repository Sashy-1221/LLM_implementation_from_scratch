import torch
import torch.nn as nn
from LLM_implementation.layer_norm import LayerNorm
from LLM_implementation.GELU_and_Feed_forward import FeedForward
from Attention.multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Multi-head self-attention layer
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Position-wise feed-forward network
        self.ff = FeedForward(cfg)

        # Layer normalization before attention and feed-forward (Pre-Norm)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout applied to residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # ---- Self-Attention Block ----
        shortcut = x                      # residual connection
        x = self.norm1(x)                 # normalize input
        x = self.att(x)                   # self-attention
        x = self.drop_shortcut(x)         # dropout
        x = x + shortcut                  # add residual

        ## here adding of shortcut directly is possible as the input
        ## output dimensions are same. As in the feedforward we do the following:
        ## x --projected to--> 4x --(GELU)--> 4x --back to--> x so no change in dimension

        # ---- Feed-Forward Block ----
        shortcut = x                      # residual connection
        x = self.norm2(x)                 # normalize input
        x = self.ff(x)                    # feed-forward network
        x = self.drop_shortcut(x)         # dropout
        x = x + shortcut                  # add residual

        return x


GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

# checking the shapes of input and output
if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)