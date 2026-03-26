import torch
import torch.nn as nn


class AdapterEncoder(nn.Module):
    def __init__(self, d, hidden=1024, ln=True):
        super().__init__()
        mods = []
        if ln:
            mods += [nn.LayerNorm(d)]
        mods += [nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)]
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x)

# --------------------------
# Simplified Encoder (Identity for pre-extracted features)
# --------------------------

class IdentityEncoder(nn.Module):
    """Identity encoder for pre-extracted features"""
    def __init__(self, d: int):
        super().__init__()
        self.d = d
    
    def forward(self, x):
        return x  # Return features as-is

