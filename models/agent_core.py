import torch
import torch.nn as nn

class AgentCore(nn.Module):
    def __init__(self, action_dim, input_dim=256, hidden_dim=128, message_dim=4):
        super(AgentCore, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # Message head (IMPORTANT)
        self.message_head = nn.Linear(hidden_dim, message_dim)

    def forward(self, h, decoded_message):
        x = torch.cat([h, decoded_message], dim=-1)

        z = self.net(x)

        action = torch.tanh(self.action_head(z))
        message = self.message_head(z)

        return action, message