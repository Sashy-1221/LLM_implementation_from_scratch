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

attn_scores = softmax(attn_scores)
print(attn_scores)
print(sum(attn_scores))  # ---> gives 1 duh

''' 
    step 3 : creating the context vector
'''

query = inputs[1] # using the same thing again

context_vec = torch.empty(query.shape)

for i,xi  in enumerate(inputs) :
    context_vec+= attn_scores[i]*xi

print(context_vec)


'''
    this is for calculating the attention scores and context vectors at once. commenting for now.
'''

# import torch
# # Input embeddings
# inputs = torch.tensor([
#     [0.43, 0.15, 0.89],  # Your
#     [0.55, 0.87, 0.66],  # journey
#     [0.57, 0.85, 0.64],  # starts
#     [0.22, 0.58, 0.33],  # with
#     [0.77, 0.25, 0.10],  # one
#     [0.05, 0.80, 0.55]   # step
# ])
#
# # Step 1: Compute attention scores
# attn_scores = inputs @ inputs.T
# print("Attention Scores:")
# print(attn_scores)
#
# # Step 2: Normalize attention scores to get attention weights
# attn_weights = torch.softmax(attn_scores, dim=-1)
# print("\nAttention Weights:")
# print(attn_weights)
#
# # Step 3: Compute context vectors
# context_vecs = attn_weights @ inputs
# print("\nContext Vectors:")
# print(context_vecs)
