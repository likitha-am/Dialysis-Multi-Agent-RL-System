import torch
import torch.nn as nn

class MessageDecoder(nn.Module):
    def __init__(self, message_dim=4, output_dim=128):
        super(MessageDecoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(message_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, m):
        return self.fc(m)