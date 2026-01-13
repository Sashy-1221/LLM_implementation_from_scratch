import torch
from math import exp

#input for the embedding

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

'''
    step 1 of embedding : computing the attn scores
    
    taking query as the inputs[1] i.e computing the attn of 2 wrt all  
'''

query = inputs[1]
attn_scores = torch.empty(inputs.shape[0])

for i,xi in enumerate(inputs):
    attn_scores[i] = torch.dot(xi,query)

print(attn_scores)

'''
    Step 2 : normalizing using naive softmax
'''

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_scores = softmax(inputs)
print(attn_scores)
print(sum(attn_scores))  # ---> gives 1 duh

# step 3 : creating the context vector