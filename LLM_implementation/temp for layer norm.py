import torch.nn as nn
import torch

torch.manual_seed(123)
batch_example = torch.randn(2, 5) # creating 2 samples of 5 values each
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) # creating a simple nn for getting a layer output
out = layer(batch_example)
print(out)

# getting the mean and variance for the layer

'''
    Using keepdim=True in operations like mean or variance calculation ensures that the
    output tensor retains the same number of dimensions as the input tensor
'''

mean = out.mean(dim=-1, keepdim=True)  # -1 refers to the last dimension
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# normalizing the layer outputs here
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

