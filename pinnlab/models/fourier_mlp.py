import torch, torch.nn as nn
from .mlp import get_act

class GaussianFF(nn.Module):
    def __init__(self, in_f, m, sigma):
        super().__init__()
        B = torch.randn(in_f, m) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        # [N,in_f] -> [N,2m]
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class FourierMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_f, out_f = cfg["in_features"], cfg["out_features"]
        H, W = cfg["hidden_layers"], cfg["hidden_width"]
        act = get_act(cfg.get("activation","tanh"))

        ff_cfg = cfg.get("fourier_features", {})
        self.use_ff = ff_cfg.get("enabled", False)
        if self.use_ff:
            m = ff_cfg.get("m", 64)
            sigma = ff_cfg.get("sigma", 10.0)
            self.ff = GaussianFF(in_f, m, sigma)
            fin = 2*m
        else:
            self.ff = nn.Identity()
            fin = in_f

        layers = []
        for _ in range(H):
            layers += [nn.Linear(fin, W), act]
            fin = W
        layers += [nn.Linear(fin, out_f)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.ff(x)
        return self.net(z)
