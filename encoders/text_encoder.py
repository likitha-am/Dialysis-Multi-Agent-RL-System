import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]