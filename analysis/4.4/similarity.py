import torch
from torch import nn

input1 = torch.load('input-1-image.pt')
input1_enc = torch.load('input-1-tensor.pt')

input2 = torch.load('input-2-image.pt')
input2_enc = torch.load('input-2-tensor.pt')

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1_enc, input2_enc)
print(f"Similarity for the output: {output}")
