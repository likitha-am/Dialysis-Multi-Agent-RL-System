import torch
import torch.nn as nn

class CategoricalEncoder(nn.Module):
    def __init__(self, num_categories, embed_dim=16):
        super(CategoricalEncoder, self).__init__()

        self.embedding = nn.Embedding(num_categories, embed_dim)

    def forward(self, x):
        return self.embedding(x).view(x.size(0), -1)