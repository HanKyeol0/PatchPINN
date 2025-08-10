import torch, torch.nn as nn
import math

def get_act(name):
    name = name.lower()
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()
    if name == "sine":
        class Sine(nn.Module):
            def forward(self, x): return torch.sin(x)
        return Sine()
    raise ValueError(f"Unknown activation {name}")

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_f, out_f = cfg["in_features"], cfg["out_features"]
        H, W = cfg["hidden_layers"], cfg["hidden_width"]
        act = get_act(cfg.get("activation","tanh"))
        layers = []
        fin = in_f
        for _ in range(H):
            layers += [nn.Linear(fin, W), act]
            fin = W
        layers += [nn.Linear(fin, out_f)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [N, in_features]
        return self.net(x)
