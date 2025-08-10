import torch, numpy as np, math
from abc import ABC, abstractmethod
from pinnlab.utils.plotting import save_plots_1d, save_plots_2d

def grads(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

class BaseExperiment(ABC):
    def __init__(self, cfg, device):
        self.cfg = cfg; self.device = device

    # ---- batch sampling ----
    @abstractmethod
    def sample_batch(self, n_f:int, n_b:int, n_0:int): ...

    # ---- losses ----
    def pde_residual_loss(self, model, batch): return torch.tensor(0., device=self.device)
    def boundary_loss(self, model, batch):     return torch.tensor(0., device=self.device)
    def initial_loss(self, model, batch):      return torch.tensor(0., device=self.device)

    # ---- eval helpers ----
    @abstractmethod
    def relative_l2_on_grid(self, model, grid_cfg): ...

    @abstractmethod
    def plot_final(self, model, grid_cfg, out_dir): ...
