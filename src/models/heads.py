import torch.nn as nn

class TextHead(nn.Module):
    def __init__(self, d_text=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_text, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.mlp(x)
