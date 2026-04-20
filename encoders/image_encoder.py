import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super(ImageEncoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        # 🔥 Lazy Linear (AUTO detects input size)
        self.fc = nn.LazyLinear(output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        return self.fc(x)