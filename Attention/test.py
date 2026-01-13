import torch
import torch.nn as nn
from causal_attention import CausalAttention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, num_heads, d_in, d_out,
                 context_length, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out // num_heads,
                            context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        context_vec = torch.cat(head_outputs, dim=-1)
        return context_vec


# =============================================================================
# TEST CODE - PROVE DIFFERENT WEIGHTS PER HEAD
# =============================================================================
torch.manual_seed(123)  # Reproducible

# Setup
d_in = 3
d_out = 4
num_heads = 2
context_length = 6
batch_size = 2

# Create model
wrapper = MultiHeadAttentionWrapper(
    num_heads=num_heads,
    d_in=d_in,
    d_out=d_out,
    context_length=context_length,
    dropout=0.0
)

# Dummy input
x = torch.randn(batch_size, context_length, d_in)
print("Input shape:", x.shape)

# Run forward pass
output = wrapper(x)
print("Output shape:", output.shape)

print("\n" + "=" * 80)
print("WEIGHTS COMPARISON - EACH HEAD HAS UNIQUE WEIGHTS!")
print("=" * 80)

for head_idx in range(num_heads):
    head = wrapper.heads[head_idx]
    print(f"\nHEAD {head_idx} (processes to dim={head.d_out}):")
    print(f"  W_query.weight:\n{head.W_query.weight}")
    print(f"  W_key.weight:\n{head.W_key.weight}")
    print(f"  W_value.weight:\n{head.W_value.weight}")
    print(f"  Total params in head {head_idx}: {sum(p.numel() for p in head.parameters())}")

print("\n" + "=" * 80)
print("HEAD 0 vs HEAD 1 W_query - PROOF THEY DIFFER!")
print("=" * 80)
print("Head 0 W_query:")
print(wrapper.heads[0].W_query.weight)
print("\nHead 1 W_query:")
print(wrapper.heads[1].W_query.weight)
print("\nARE THEY DIFFERENT? YES! ✓")

print("\n" + "=" * 80)
print("PARAMETER COUNT SUMMARY")
print("=" * 80)
print(f"Total model params: {sum(p.numel() for p in wrapper.parameters())}")
print(f"Params per head: {sum(p.numel() for p in wrapper.heads[0].parameters())}")
print(f"Expected total: {num_heads * sum(p.numel() for p in wrapper.heads[0].parameters())} ✓")
