import math
import numpy as np
import torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class ReactionDiffusion2D(BaseExperiment):
    """
    Linear RD in 2D: u_t - D (u_xx + u_yy) - r u = 0 on [0,1]x[0,1].
    Manufactured: u*(x,y,t) = sin(pi x) sin(pi y) * exp(-λ t).
    residual = (-λ + D*2*pi^2 - r) u*. If λ = D*2*pi^2 - r, residual==0.
    Dirichlet BC from u*, IC at t=t0.
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]; ya, yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]
        self.rect = Rectangle(xa, xb, ya, yb, device)
        self.D = float(cfg.get("D", 1.0))
        self.r = float(cfg.get("r", 0.0))
        lam_cfg = cfg.get("lambda", "auto")
        self.lmbda = (self.D * 2.0 * (math.pi**2) - self.r) if (isinstance(lam_cfg, str) and lam_cfg.lower()=="auto") else float(lam_cfg)
        self.input_dim = 3  # (x,y,t)

    def u_star(self, x, y, t):
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.exp(-self.lmbda * t)

    def sample_batch(self, n_f, n_b, n_0):
        device = self.rect.device
        # interior (x,y,t)
        XY = self.rect.sample(n_f)
        t = torch.rand(n_f,1,device=device)*(self.t1-self.t0)+self.t0
        X_f = torch.cat([XY, t], 1)

        # Dirichlet boundary on all edges with random t
        nb = max(1, n_b//4)
        xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
        y = torch.rand(nb,1,device=device)*(yb-ya)+ya
        x = torch.rand(nb,1,device=device)*(xb-xa)+xa
        top    = torch.cat([x, torch.full_like(x, yb)], 1)
        bottom = torch.cat([x, torch.full_like(x, ya)], 1)
        left   = torch.cat([torch.full_like(y, xa), y], 1)
        right  = torch.cat([torch.full_like(y, xb), y], 1)
        X_b_spatial = torch.cat([top,bottom,left,right], 0)
        t_b = torch.rand(X_b_spatial.size(0),1,device=device)*(self.t1-self.t0)+self.t0
        X_b = torch.cat([X_b_spatial, t_b], 1)
        u_b = self.u_star(X_b[:,0:1], X_b[:,1:2], X_b[:,2:3])

        # IC at t=t0
        XY0 = self.rect.sample(n_0)
        t0 = torch.full((n_0,1), self.t0, device=device)
        X_0 = torch.cat([XY0, t0], 1)
        u0 = self.u_star(X_0[:,0:1], X_0[:,1:2], X_0[:,2:3])

        return {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}

    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])
        u = model(X)
        du = grad_sum(u, X)
        u_x, u_y, u_t = du[:,0:1], du[:,1:2], du[:,2:3]
        d2ux = grad_sum(u_x, X); d2uy = grad_sum(u_y, X)
        u_xx, u_yy = d2ux[:,0:1], d2uy[:,1:2]
        res = u_t - self.D * (u_xx + u_yy) - self.r * u
        return res.pow(2)

    def boundary_loss(self, model, batch):
        pred = model(batch["X_b"])
        return (pred - batch["u_b"]).pow(2)

    def initial_loss(self, model, batch):
        pred = model(batch["X_0"])
        return (pred - batch["u0"]).pow(2)

    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
        idxs = [0, nt//2, nt-1] if nt>=3 else list(range(nt))
        rels = []
        with torch.no_grad():
            for ti in idxs:
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], 1)
                U_pred = model(XYT).reshape(nx, ny)
                U_true = self.u_star(Xg, Yg, T)
                rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
                rels.append(rel.item())
        return float(np.mean(rels))

    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
        figs = {}
        with torch.no_grad():
            for label, ti in zip(["t0","tmid","t1"], [0, nt//2, nt-1]):
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], 1)
                U_pred = model(XYT).reshape(nx, ny).cpu().numpy()
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()
                out = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"rd2d_{label}")
                figs.update(out)
        return figs
