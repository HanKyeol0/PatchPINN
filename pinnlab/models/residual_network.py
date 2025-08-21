import torch
import torch.nn as nn
import torch.autograd as autograd

def get_act(name):
    name = name.lower()
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()
    if name == "sine":
        class Sine(nn.Module):
            def forward(self, x): return torch.sin(x)
        return Sine()
    raise ValueError(f"Unknown activation {name}")

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super().__init__()
        self.main = nn.Sequential(                 # <-- central path
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim)  # Projection shortcut

    def forward(self, x):
        return self.main(x) + self.shortcut(x)

class ResidualNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim, hidden_dim, output_dim = cfg["in_features"], cfg["hidden_dim"], cfg["out_features"]
        num_blocks = cfg.get("num_blocks", 4)
        act = get_act(cfg.get("activation", "tanh"))
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualBlock(input_dim, hidden_dim, hidden_dim, act))
        for _ in range(num_blocks - 2):
            self.blocks.append(ResidualBlock(hidden_dim, hidden_dim, hidden_dim, act))
        self.blocks.append(ResidualBlock(hidden_dim, hidden_dim, output_dim, act))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x