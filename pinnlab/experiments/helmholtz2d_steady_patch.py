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

    def _predict_grid_via_patches(self, model, nx: int, ny: int):
        """
        Produce a dense prediction U_pred[nx, ny] by sliding fixed-size patches
        of points and averaging overlaps. Works even when (nx-1) % px != 0 or
        (ny-1) % py != 0 by clamping indices at the domain boundary.

        Model is called on tensors shaped [P,2] where P=(px+1)*(py+1).
        """
        device = self.device
        # Make evaluation grid coordinates
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, device)  # [nx,ny] each
        xs = Xg[:, 0]  # xs[i] == Xg[i,0]
        ys = Yg[0, :]  # ys[j] == Yg[0,j]

        px, py = self.px, self.py             # in *cells*
        gx, gy = nx - 1, ny - 1               # number of cells along each axis
        # Tile starts in *cell* indices (stride = patch size)
        starts_x = list(range(0, gx + 1, px))
        starts_y = list(range(0, gy + 1, py))

        # Accumulators for overlap-averaging
        S = torch.zeros(nx, ny, device=device)   # sum of predictions
        W = torch.zeros(nx, ny, device=device)   # counts (weights)

        with torch.no_grad():
            for ix0 in starts_x:
                # indices in *point* space for this patch (clamped to grid)
                ix_vec = torch.arange(ix0, ix0 + px + 1, device=device)
                ix_vec = ix_vec.clamp_(0, gx)  # length px+1
                x_coords = xs[ix_vec]          # [px+1]

                for iy0 in starts_y:
                    iy_vec = torch.arange(iy0, iy0 + py + 1, device=device)
                    iy_vec = iy_vec.clamp_(0, gy)  # length py+1
                    y_coords = ys[iy_vec]          # [py+1]

                    XX, YY = torch.meshgrid(x_coords, y_coords, indexing="ij")  # [px+1, py+1]
                    coords = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)  # [P,2]

                    # Model expects a fixed-size patch [P,2]
                    U_patch = model(coords).reshape(px + 1, py + 1).squeeze(-1)   # [px+1, py+1]

                    # Scatter-add into the global grid (average overlaps)
                    # (Handles clamped duplicates at edges automatically via W.)
                    for a in range(px + 1):
                        ix = int(ix_vec[a].item())
                        for b in range(py + 1):
                            iy = int(iy_vec[b].item())
                            S[ix, iy] += U_patch[a, b]
                            W[ix, iy] += 1.0

        # Avoid divide-by-zero (shouldn't happen, but be safe)
        W = torch.where(W > 0, W, torch.ones_like(W))
        U_pred = S / W  # [nx, ny]
        return Xg, Yg, U_pred

    # ---- evaluation: relative L2 using patch-based inference ----
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg, U_pred = self._predict_grid_via_patches(model, nx, ny)
        with torch.no_grad():
            U_true = self.u_star(Xg, Yg)                 # [nx,ny]
            num = torch.linalg.norm((U_pred - U_true).reshape(-1))
            den = torch.linalg.norm(U_true.reshape(-1))
            return (num / den).item()

    # ---- plotting: also using patch-based inference ----
    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg, U_pred = self._predict_grid_via_patches(model, nx, ny)
        with torch.no_grad():
            U_true = self.u_star(Xg, Yg)
        return save_plots_2d(
            Xg.detach().cpu().numpy(),
            Yg.detach().cpu().numpy(),
            U_true.detach().cpu().numpy(),
            U_pred.detach().cpu().numpy(),
            out_dir,
            "helmholtz2d_steady_patch"
        )
