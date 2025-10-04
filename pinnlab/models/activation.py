import torch, torch.nn as nn

def get_act(name):
    name = name.lower()
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()
    if name == "sine":
        class Sine(nn.Module):
            def forward(self, x): return torch.sin(x)
        return Sine()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "softplus":
        return nn.Softplus(beta=1.0)
    raise ValueError(f"Unknown activation {name}")