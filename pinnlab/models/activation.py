import torch, torch.nn as nn

def get_act(name):
    name = name.lower()
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()
    if name == "sine":
        class Sine(nn.Module):
            def forward(self, x): return torch.sin(x)
        return Sine()
    raise ValueError(f"Unknown activation {name}")