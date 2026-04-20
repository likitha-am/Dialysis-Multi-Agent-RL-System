import torch
import torch.nn as nn

class MessageEncoder(nn.Module):
    def __init__(self, input_dim=128, message_dim=4):
        super(MessageEncoder, self).__init__()

        self.fc = nn.Linear(input_dim, message_dim)

    def forward(self, h):
        return self.fc(h)