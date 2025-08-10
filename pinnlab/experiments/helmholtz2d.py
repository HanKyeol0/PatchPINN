import torch, math, numpy as np
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class Helmholtz2D(BaseExperiment):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]; ya, yb = cfg["domain"]["y"]
        self.rect = Rectangle(xa, xb, ya, yb, device)
        self.k = float(cfg.get("k", 2.0))

    def u_star(self, x, y): return torch.sin(math.pi*x) * torch.sin(math.pi*y)
    def f(self, x, y):     return (2*math.pi**2 - self.k**2)*self.u_star(x,y)

    def sample_batch(self, n_f, n_b, n_0):
        X_f = self.rect.sample(n_f)
        nb = n_b//4
        xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
        y = torch.rand(nb,1,device=self.rect.device)*(yb-ya)+ya
        x = torch.rand(nb,1,device=self.rect.device)*(xb-xa)+xa
        top    = torch.cat([x, torch.full_like(x, yb)], 1)
        bottom = torch.cat([x, torch.full_like(x, ya)], 1)
        left   = torch.cat([torch.full_like(y, xa), y], 1)
        right  = torch.cat([torch.full_like(y, xb), y], 1)
        X_b = torch.cat([top,bottom,left,right], dim=0)
        u_b = self.u_star(X_b[:,0:1], X_b[:,1:2])
        return {"X_f": X_f, "X_b": X_b, "u_b": u_b}

    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])
        u = model(X)
        du = grad_sum(u, X)         # [N,2]
        u_x, u_y = du[:,0:1], du[:,1:2]
        d2ux = grad_sum(u_x, X)     # [N,2]
        d2uy = grad_sum(u_y, X)
        u_xx, u_yy = d2ux[:,0:1], d2uy[:,1:2]
        x, y = X[:,0:1], X[:,1:2]
        res = u_xx + u_yy + (self.k**2)*u - self.f(x,y)
        return res.pow(2)

    def boundary_loss(self, model, batch):
        Xb, ub = batch["X_b"], batch["u_b"]
        pred = model(Xb)
        return (pred - ub).pow(2)

    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        X, Y = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
        with torch.no_grad():
            U = model(XY).reshape(nx,ny)
        U_true = self.u_star(X,Y)
        rel = torch.linalg.norm((U - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
        return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        X, Y = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XY).reshape(nx,ny).cpu().numpy()
        U_true = self.u_star(X,Y).cpu().numpy()
        from pinnlab.utils.plotting import save_plots_2d
        return save_plots_2d(X.cpu().numpy(), Y.cpu().numpy(), U_true, U_pred, out_dir, "helmholtz2d")
