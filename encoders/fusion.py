import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(FusionLayer, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=-1)
        return self.fc(x)