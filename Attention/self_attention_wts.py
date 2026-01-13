import torch.nn as nn
import torch
class self_attention(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.wk = nn.Parameter(torch.rand(d_in, d_out))
        self.wq = nn.Parameter(torch.rand(d_in, d_out))
        self.wv = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x): # x is the input
        keys = x @ self.wk
        values = x @ self.wv
        queries = x @ self.wq
        attn_scores = keys @ values.T  # ---> multiplying with transpose to get the dot product

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )



'''
    another implementation option can be that instead of manually implementing torch.rand 
    we can use "" def __init__(self, d_in, d_out, qkv_bias=False): and self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) ""
    
'''