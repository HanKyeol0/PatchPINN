import math
import numpy as np
import torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d
from pinnlab.data.samplers import sample_patches_2d_steady
from pinnlab.utils.plotting import save_plots_2d

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
        return {"X_f": x_f, "X_b": x_b, "u_b": u_b}
    
    def pde_residual_loss(self, model, batch):
        errs = []
        device = self.device

        for coords in batch["X_f"]:
            if coords.numel() == 0:
                continue
            X = make_leaf(coords)
            u = model(X)
            du = grad_sum(u, X)
            u_x, u_y = du[:, 0:1], du[:, 1:2]
            d2ux = grad_sum(u_x, X)
            d2uy = grad_sum(u_y, X)
            u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
            res_pred = u_xx + u_yy + self.lam * u
            f_xy = self.f(X[:, 0:1], X[:, 1:2])
            return (res_pred - f_xy).pow(2)
        
        for patch in batch["X_b"]:
            coords = patch["coords"]
            mask = patch["boundary_mask"]

            if (~mask).any():
                X_int = make_leaf(coords[~mask])
                u = model(X_int)
                du = grad_sum(u, X_int)
                u_x, u_y = du[:, 0:1], du[:, 1:2]
                d2ux = grad_sum(u_x, X_int)
                d2uy = grad_sum(u_y, X_int)
                u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
                res_pred = u_xx + u_yy + self.lam * u
                f_xy = self.f(X_int[:, 0:1], X_int[:, 1:2])
                errs.append((res_pred - f_xy).pow(2))
            
            if len(errs) == 0:
                return torch.tensor(0.0, device=device)
            
            return torch.cat(errs, dim=0)
    
    def boundary_loss(self, model, batch):
        device = self.device
        boundary_patches = batch["X_b"]
        u_b_list = batch["u_b"]

        errs = []
        for i, patch in enumerate(boundary_patches):
            coords = patch["coords"]
            mask = patch["boundary_mask"]
            if not mask.any():
                continue
            model(coords)
            Xb = coords[mask]
            pred = model(Xb)

            ub_full = u_b_list[i]
            if ub_full.dim() == 1:
                ub_full = ub_full[:, None]
            ub = ub_full[mask]

            if torch.isnan(ub).any():
                valid = ~torch.isnan(ub.squeeze(1))
                if valid.any():
                    errs.append((pred[valid] - ub[valid]).pow(2))
            else:
                errs.append((pred-ub).power(2))
    
        if len(errs) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.cat(errs,dim=0)

    def initial_loss(self, model, batch):
        return torch.tensor(0.0, device=self.device)
    
    # ---- eval & plot (same spirit as point-wise version) ----
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, self.device)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XY).reshape(nx, ny)
            U_true = self.u_star(Xg, Yg)
        rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
        return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, self.device)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XY).reshape(nx, ny).cpu().numpy()
            U_true = self.u_star(Xg, Yg).cpu().numpy()
        return save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred,
                             out_dir, "helmholtz2d_steady_patch")