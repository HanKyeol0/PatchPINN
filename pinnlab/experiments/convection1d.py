import math
import numpy as np
import torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Interval

class Convection1D(BaseExperiment):
    """
    u_t + c u_x = 0 on x in [0,1], periodic BC in x.
    Manufactured solution: u*(x,t) = sin(2π (x - c t)), which is 1-periodic in x.
    Then u_t + c u_x = 0 exactly (residual = 0).
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]
        self.t0, self.t1 = cfg["domain"]["t"]
        self.c = float(cfg.get("c", 1.0))
        self.x_dom = Interval(xa, xb, device)
        self.input_dim = 2  # (x,t)

    # analytic
    def u_star(self, x, t):
        return torch.sin(2.0 * math.pi * (x - self.c * t))

    # sampling
    def sample_batch(self, n_f, n_b, n_0):
        # interior (x,t)
        x = self.x_dom.sample(n_f)
        t = torch.rand(n_f,1,device=self.x_dom.device)*(self.t1-self.t0)+self.t0
        X_f = torch.cat([x,t], dim=1)

        # periodic BC: x=xa vs x=xb at same random t
        nb = max(1, n_b)
        t_b = torch.rand(nb,1,device=self.x_dom.device)*(self.t1-self.t0)+self.t0
        xL = torch.full((nb,1), self.x_dom.a, device=self.x_dom.device)
        xR = torch.full((nb,1), self.x_dom.b, device=self.x_dom.device)
        X_bp1 = torch.cat([xL, t_b], 1)
        X_bp2 = torch.cat([xR, t_b], 1)

        # initial condition t=t0
        x0 = self.x_dom.sample(n_0)
        t0 = torch.full((n_0,1), self.t0, device=self.x_dom.device)
        X_0 = torch.cat([x0,t0], 1)
        u0 = self.u_star(x0, t0)

        return {"X_f": X_f, "X_bp1": X_bp1, "X_bp2": X_bp2, "X_0": X_0, "u0": u0}

    # losses
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])   # [N,2] (x,t)
        u = model(X)
        du = grad_sum(u, X)
        u_x, u_t = du[:,0:1], du[:,1:2]
        res = u_t + self.c * u_x
        return res.pow(2)

    def boundary_loss(self, model, batch):
        A = model(batch["X_bp1"])  # x=xa
        B = model(batch["X_bp2"])  # x=xb
        return (A - B).pow(2)      # periodic equality

    def initial_loss(self, model, batch):
        pred = model(batch["X_0"])
        return (pred - batch["u0"]).pow(2)

    # eval/plots
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, nt = grid_cfg["nx"], grid_cfg["nt"]
        xs = torch.linspace(self.x_dom.a, self.x_dom.b, nx, device=self.x_dom.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.x_dom.device)
        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XT).reshape(nx, nt)
        U_true = self.u_star(X, T)
        rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
        return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_1d
        nx, nt = grid_cfg["nx"], grid_cfg["nt"]
        xs = torch.linspace(self.x_dom.a, self.x_dom.b, nx, device=self.x_dom.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.x_dom.device)
        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XT).reshape(nx, nt).cpu().numpy()
        U_true = self.u_star(X, T).cpu().numpy()
        return save_plots_1d(xs.cpu().numpy(), ts.cpu().numpy(), U_true, U_pred, out_dir, "convection1d")
