import math
import numpy as np
import torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d
from pinnlab.data.samplers import sample_patches_2d_steady

class Helmholtz2dSteady_patch(BaseExperiment):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.xa, self.xb = cfg["domain"]["x"]
        self.ya, self.yb = cfg["domain"]["y"]

        self.px = cfg["patch"]["x"] # patch 크기
        self.py = cfg["patch"]["y"]

        self.gx = cfg["grid"]["x"] # grid 격자 개수 (100이면 x축을 100개로 나누는 것)
        self.gy = cfg["grid"]["y"]

        self.a1 = float(cfg.get("a1", 1.0))
        self.a2 = float(cfg.get("a2", 4.0))
        self.lam = float(cfg.get("lambda", 1.0))
        self.input_dim = 2  # (x,y)

    def u_star(self, x, y):
        return torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y)
    
    def f(self, x, y):
        coeff = (-(self.a1**2 + self.a2**2) * (math.pi**2) + self.lam)
        return coeff * self.u_star(x, y)

    def sample_patches(self):
        # boundary condition
        def g_dirichlet(x, y):
            # Example boundary condition u = sin(pi x) + cos(pi y)
            import math
            return torch.sin(math.pi * x) + torch.cos(math.pi * y)

        x_f, x_b, u_b = sample_patches_2d_steady(
            self.xa, self.xb,
            self.ya, self.yb,
            self.px, self.py,
            self.gx, self.gy,
            device="cuda" if torch.cuda.is_available() else "cpu",
            stride_x=1, stride_y=1,
            boundary_fn=g_dirichlet,
        )
        
        return x_f, x_b, u_b