import torch
import torch.nn as nn

class NumericEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=32):
        super(NumericEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)