import torch

inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]  # step (x^6)
])

x2 = inputs[1]

# defining input and output dimensions
d_in, d_out = inputs.shape[1], 2

torch.manual_seed(123)
wq = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
wv = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
wk = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

q2 = x2 @ wq
k2 = x2 @ wk
v2 = x2 @ wv
print(q2)

keys = inputs @ wk
values = inputs @ wv

keys_2 = keys[1]
attn_score_22 = q2.dot(keys_2)
print(attn_score_22) # getting attn score of only 2 with 2

attn_scores = q2 @ keys.T
print(attn_scores) # mat mul to get all the attn scores

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores / d_k**0.5, dim=-1)
print(attn_weights_2) # normailzed

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

'''

# ========================================
# DEBUG: Print All Shapes and Values
# ========================================
print("\n" + "="*60)
print("DEBUG: ALL VARIABLE SHAPES & VALUES")
print("="*60)

debug_vars = {
    'inputs': inputs,
    'x2': x2,
    'd_in': d_in,
    'd_out': d_out,
    'wq': wq,
    'wk': wk,
    'wv': wv,
    'q2': q2,
    'k2': k2,
    'v2': v2,
    'keys': keys,
    'values': values,
    'keys_2': keys_2,
    'attn_score_22': attn_score_22,
    'attn_scores': attn_scores,
    'd_k': d_k,
    'attn_weights_2': attn_weights_2,
    'context_vec_2': context_vec_2
}

for name, var in debug_vars.items():
    if torch.is_tensor(var):
        print(f"{name:<15} | Shape: {str(var.shape):<12} | Values: {var}")
    else:
        print(f"{name:<15} | Type: {type(var)}      | Value: {var}")

print("\n" + "="*60)
print("DIMENSION CHECKS")
print("="*60)
print(f"Matrix multiplication checks:")
print(f"  x2 @ wq:     {x2.shape} @ {wq.shape} → {q2.shape} ✓")
print(f"  inputs @ wk: {inputs.shape} @ {wk.shape} → {keys.shape} ✓")
print(f"  q2 @ keys.T: {q2.shape} @ {keys.T.shape} → {attn_scores.shape} ✓")
print(f"  attn_weights_2 @ values: {attn_weights_2.shape} @ {values.shape} → {context_vec_2.shape} ✓")

print(f"\nAttention weights sum to 1.0: {attn_weights_2.sum():.1f} ✓")
print(f"Final context vector shape matches d_out: {context_vec_2.shape} == ({d_out},) ✓")

'''



