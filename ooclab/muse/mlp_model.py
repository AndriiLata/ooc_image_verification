import torch, torch.nn as nn, torch.nn.functional as F

class MUSEMLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.GELU(),
            nn.Linear(hid_dim, 1)
        )
    def forward(self, x):            # x: [B, in_dim]
        return self.net(x).squeeze(1) # logits
