import math
import torch
import numpy as np

from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class Helmholtz2D_Patch(BaseExperiment):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]
        ya, yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]

        self.rect = Rectangle(xa, xb, ya, yb, device)
        self.c = float(cfg.get("c", 1.0))

        

        omega_cfg = cfg.get("omega", "auto")
        if isinstance(omega_cfg, str) and omega_cfg.lower() == "auto":
            self.omega = math.sqrt(2.0) * math.pi * self.c
        else:
            self.omega = float(omega_cfg)

    def u_star(self, x, y, t):
        return torch.sin(math.pi * x) * torch