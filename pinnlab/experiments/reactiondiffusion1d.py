import math
import numpy as np
import torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Interval

class ReactionDiffusion1D(BaseExperiment):
    """
    Linear RD in 1D: u_t - D u_xx - r u = 0 on x in [0,1].
    Manufactured solution: u*(x,t) = sin(pi x) * exp(-λ t).
    Then residual = (-λ + D*pi^2 - r) u*. If λ = D*pi^2 - r, residual==0.
    Dirichlet BC from u*, IC at t=t0 from u*.
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]
        self.t0, self.t1 = cfg["domain"]["t"]
        self.x_dom = Interval(xa, xb, device)
        self.D = float(cfg.get("D", 1.0))
        self.r = float(cfg.get("r", 0.0))
        lam_cfg = cfg.get("lambda", "auto")
        self.lmbda = (self.D * (math.pi**2) - self.r) if (isinstance(lam_cfg, str) and lam_cfg.lower()=="auto") else float(lam_cfg)
        self.input_dim = 2  # (x,t)

    def u_star(self, x, t):
        return torch.sin(math.pi * x) * torch.exp(-self.lmbda * t)

    def sample_batch(self, n_f, n_b, n_0):
        # interior
        x = self.x_dom.sample(n_f)
        t = torch.rand(n_f,1,device=self.x_dom.device)*(self.t1-self.t0)+self.t0
        X_f = torch.cat([x,t], 1)

        # Dirichlet on x=a and x=b
        nb = max(1, n_b//2)
        t_b = torch.rand(2*nb,1,device=self.x_dom.device)*(self.t1-self.t0)+self.t0
        xa = torch.full((nb,1), self.x_dom.a, device=self.x_dom.device)
        xb = torch.full((nb,1), self.x_dom.b, device=self.x_dom.device)
        Xb = torch.cat([torch.cat([xa, t_b[:nb]],1), torch.cat([xb, t_b[nb:]],1)], dim=0)
        u_b = self.u_star(Xb[:,0:1], Xb[:,1:2])

        # IC
        x0 = self.x_dom.sample(n_0)
        t0 = torch.full((n_0,1), self.t0, device=self.x_dom.device)
        X_0 = torch.cat([x0,t0], 1)
        u0 = self.u_star(x0, t0)

        return {"X_f": X_f, "X_b": Xb, "u_b": u_b, "X_0": X_0, "u0": u0}

    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])
        u = model(X)
        du = grad_sum(u, X)
        u_x, u_t = du[:,0:1], du[:,1:2]
        d2ux = grad_sum(u_x, X)
        u_xx = d2ux[:,0:1]
        res = u_t - self.D * u_xx - self.r * u
        return res.pow(2)

    def boundary_loss(self, model, batch):
        pred = model(batch["X_b"])
        return (pred - batch["u_b"]).pow(2)

    def initial_loss(self, model, batch):
        pred = model(batch["X_0"])
        return (pred - batch["u0"]).pow(2)

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
        return save_plots_1d(xs.cpu().numpy(), ts.cpu().numpy(), U_true, U_pred, out_dir, "rd1d")
