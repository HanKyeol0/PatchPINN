import torch, numpy as np, math
from pinnlab.experiments.base import BaseExperiment, grads
from pinnlab.data.geometries import Rectangle, linspace_2d

class Poisson2D(BaseExperiment):
    """
    -∇²u = f,  Dirichlet from u*(x,y)=sin(pi x) sin(pi y)
    => ∇²u* = -2pi^2 u*, so f = 2pi^2 u*
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]; ya, yb = cfg["domain"]["y"]
        self.rect = Rectangle(xa, xb, ya, yb, device)

    def u_star(self, x, y):
        return torch.sin(math.pi*x) * torch.sin(math.pi*y)

    def f(self, x, y):
        return 2*(math.pi**2)*self.u_star(x,y)

    def sample_batch(self, n_f, n_b, n_0):
        X_f = self.rect.sample(n_f)
        # boundary (same policy as Helmholtz)
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
        X = batch["X_f"].requires_grad_(True)
        x, y = X[:,0:1], X[:,1:2]
        u = model(X)
        u_x = grads(u, x); u_y = grads(u, y)
        u_xx = grads(u_x, x); u_yy = grads(u_y, y)
        res = -(u_xx + u_yy) - self.f(x,y)
        return (res**2)

    def boundary_loss(self, model, batch):
        Xb, ub = batch["X_b"], batch["u_b"]
        pred = model(Xb)
        return (pred - ub)**2

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
        return save_plots_2d(X.cpu().numpy(), Y.cpu().numpy(), U_true, U_pred, out_dir, "poisson2d")
