import math
import numpy as np
import torch
from pinnlab.experiments.base_patch import BaseExperiment_Patch, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d
from pinnlab.data.samplers import sample_patches_2d_steady
from pinnlab.utils.plotting import save_plots_2d

# ---- local safe grad helpers ----
def _grad_sum_allow_unused(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Safe dy/dx via sum-of-outputs trick."""
    (g,) = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    if g is None:
        g = x * 0  # zero with dependency on x
    return g

def _ensure_leaf_requires_grad(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach().requires_grad_(True)

class Helmholtz2DSteady_patch(BaseExperiment_Patch):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.xa, self.xb = cfg["domain"]["x"]
        self.ya, self.yb = cfg["domain"]["y"]

        self.px = cfg["patch"]["x"]  # number of points in x per patch
        self.py = cfg["patch"]["y"]  # number of points in y per patch
        self.patch_size = self.px * self.py  # Total points per patch

        self.gx = cfg["grid"]["x"]  # total grid points in x
        self.gy = cfg["grid"]["y"]  # total grid points in y

        self.a1 = float(cfg.get("a1", 1.0))
        self.a2 = float(cfg.get("a2", 4.0))
        self.lam = float(cfg.get("lambda", 1.0))
        self.input_dim = 2  # (x,y)

        self.batch_size = cfg.get("batch_size", 32)

    def _iterate_in_chunks(self, lst, chunk):
        for i in range(0, len(lst), chunk):
            yield i, lst[i:i+chunk]

    def _gather_residual_patches(self, batch):
        """Build lists for residual computation."""
        coords_list, keep_mask_list = [], []
        P = self.patch_size
        device = self.device

        # 1) pure interior patches -> keep all points
        for coords in batch.get("X_f", []) or []:
            assert coords.shape[0] == P, f"Expected P={P}, got {coords.shape[0]}"
            coords_list.append(coords)
            keep_mask_list.append(torch.ones(P, dtype=torch.bool, device=device))

        # 2) boundary patches -> keep only interior points
        for patch in batch.get("X_b", []) or []:
            coords = patch["coords"]
            bmask = patch["boundary_mask"]
            assert coords.shape[0] == P, f"Expected P={P}, got {coords.shape[0]}"
            coords_list.append(coords)
            keep_mask_list.append(~bmask)  # keep interior rows only

        return coords_list, keep_mask_list

    def _gather_boundary_patches(self, batch):
        """Build lists for boundary loss."""
        coords_list, bmask_list, ub_list = [], [], []
        if batch.get("X_b", None) is None:
            return coords_list, bmask_list, ub_list
        for i, patch in enumerate(batch["X_b"]):
            coords = patch["coords"]
            bmask = patch["boundary_mask"]
            coords_list.append(coords)
            bmask_list.append(bmask)
            if batch.get("u_b", None) is not None and i < len(batch["u_b"]) and batch["u_b"][i] is not None:
                ub = batch["u_b"][i]
                if ub.dim() == 1:
                    ub = ub[:, None]
                ub_list.append(ub)
            else:
                ub_list.append(None)
        return coords_list, bmask_list, ub_list

    def u_star(self, x, y):
        """Exact solution for the Helmholtz equation."""
        return torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y)
    
    def f(self, x, y):
        """Source term for the Helmholtz equation."""
        coeff = (-(self.a1**2 + self.a2**2) * (math.pi**2) + self.lam)
        return coeff * self.u_star(x, y)

    def sample_patches(self):
        # Use exact solution as boundary condition for testing
        def g_dirichlet(x, y):
            return self.u_star(x.squeeze(-1), y.squeeze(-1))

        out = sample_patches_2d_steady(
            self.xa, self.xb,
            self.ya, self.yb,
            self.px, self.py,  # patch size in points
            self.gx, self.gy,  # total grid points
            device="cuda" if torch.cuda.is_available() else "cpu",
            stride_x=None,  # Will default to px (non-overlapping)
            stride_y=None,  # Will default to py (non-overlapping)
            boundary_fn=g_dirichlet,
        )
        x_f, x_b, u_b = out["interior_patches"], out["boundary_patches"], out["true_boundary"]
        return x_f, x_b, u_b
    
    def sample_batch(self, *_, **__):
        x_f, x_b, u_b = self.sample_patches()

        assert isinstance(x_f, list) and (len(x_f) == 0 or torch.is_tensor(x_f[0])), \
            f"X_f must be List[Tensor], got {type(x_f)}"
        assert isinstance(x_b, list) and (len(x_b) == 0 or isinstance(x_b[0], dict)), \
            f"X_b must be List[Dict], got {type(x_b)}"
        if u_b is not None:
            assert isinstance(u_b, list), f"u_b must be List[Tensor] or None, got {type(u_b)}"

        return {"X_f": x_f, "X_b": x_b, "u_b": u_b}
    
    def pde_residual_loss(self, model, batch):
        device = self.device
        P = self.patch_size

        coords_list, keep_mask_list = self._gather_residual_patches(batch)
        if len(coords_list) == 0:
            return torch.tensor(0.0, device=device)

        errs = []
        bs = self.batch_size

        for start, coords_chunk in self._iterate_in_chunks(coords_list, bs):
            k_masks_chunk = keep_mask_list[start:start+len(coords_chunk)]

            C = torch.stack(coords_chunk, dim=0).to(device)  # [B,P,2]
            K = torch.stack(k_masks_chunk, dim=0).to(device)  # [B,P]
            B = C.size(0)

            X = _ensure_leaf_requires_grad(C)  # leaf [B,P,2]
            U = model(X).reshape(B, P, 1)  # [B,P,1]

            # flatten for faster autograd calls
            U_flat = U.view(B * P, 1)  # [BP,1]
            X_flat = X.view(B * P, 2)  # [BP,2]

            # first derivatives
            dU = _grad_sum_allow_unused(U_flat, X_flat).view(B, P, 2)  # [B,P,2]
            u_x = dU[..., 0:1]  # [B,P,1]
            u_y = dU[..., 1:2]  # [B,P,1]

            # second derivatives
            d2ux = _grad_sum_allow_unused(u_x.view(B * P, 1), X_flat).view(B, P, 2)
            d2uy = _grad_sum_allow_unused(u_y.view(B * P, 1), X_flat).view(B, P, 2)
            u_xx = d2ux[..., 0:1]  # [B,P,1]
            u_yy = d2uy[..., 1:2]  # [B,P,1]

            # PDE residual: ∇²u + λu = f
            res_pred = (u_xx + u_yy + self.lam * U).squeeze(-1)  # [B,P]
            f_xy = self.f(X[..., 0], X[..., 1])  # [B,P]

            diff = (res_pred - f_xy)[K]  # 1-D masked
            errs.append(diff.pow(2).unsqueeze(1))  # [N_kept,1]

        return torch.cat(errs, dim=0)

    def initial_loss(self, model, batch):
        """No initial condition for steady-state problems."""
        return torch.tensor(0.0, device=self.device)
    
    def boundary_loss(self, model, batch):
        device = self.device

        coords_list, bmask_list, ub_list = self._gather_boundary_patches(batch)
        if len(coords_list) == 0:
            return torch.tensor(0.0, device=device)

        errs = []
        bs = self.batch_size

        for start, coords_chunk in self._iterate_in_chunks(coords_list, bs):
            bmask_chunk = bmask_list[start:start+len(coords_chunk)]
            ub_chunk = ub_list[start:start+len(coords_chunk)]

            C = torch.stack(coords_chunk, dim=0).to(device)  # [B,P,2]
            B, P, _ = C.shape

            with torch.set_grad_enabled(False):
                pred_full = model(C).squeeze(-1)  # [B,P]

            for i in range(B):
                mask_i = bmask_chunk[i].to(device)  # [P]
                if ub_chunk[i] is not None:
                    ub_full = ub_chunk[i].to(device).squeeze(-1)  # [P]
                    ub_b = ub_full[mask_i]  # [Nb]
                else:
                    # Use exact solution as boundary condition
                    Xi = C[i]
                    ub_b = self.u_star(Xi[:, 0], Xi[:, 1])[mask_i]  # [Nb]
                pred_b = pred_full[i][mask_i]  # [Nb]
                
                # Drop NaNs defensively
                if torch.isnan(ub_b).any():
                    valid = ~torch.isnan(ub_b)
                    if valid.any():
                        errs.append((pred_b[valid] - ub_b[valid]).pow(2).unsqueeze(1))
                else:
                    errs.append((pred_b - ub_b).pow(2).unsqueeze(1))

        return torch.cat(errs, dim=0) if len(errs) > 0 else torch.tensor(0.0, device=device)

    def _predict_grid_via_patches(self, model, nx: int, ny: int):
        """Produce a dense prediction by sliding fixed-size patches."""
        device = self.device
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, device)
        xs = Xg[:, 0]
        ys = Yg[0, :]

        px, py = self.px, self.py  # points per patch
        
        # Tile starts (in point indices)
        starts_x = list(range(0, nx - px + 1, px))
        starts_y = list(range(0, ny - py + 1, py))
        
        # Add final patches to cover the entire domain
        if len(starts_x) == 0 or starts_x[-1] + px < nx:
            starts_x.append(max(0, nx - px))
        if len(starts_y) == 0 or starts_y[-1] + py < ny:
            starts_y.append(max(0, ny - py))

        S = torch.zeros(nx, ny, device=device)
        W = torch.zeros(nx, ny, device=device)

        with torch.no_grad():
            for ix0 in starts_x:
                ix_vec = torch.arange(ix0, min(ix0 + px, nx), device=device)
                x_coords = xs[ix_vec]

                for iy0 in starts_y:
                    iy_vec = torch.arange(iy0, min(iy0 + py, ny), device=device)
                    y_coords = ys[iy_vec]

                    XX, YY = torch.meshgrid(x_coords, y_coords, indexing="ij")
                    coords = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)

                    # Model expects fixed-size patch [px*py, 2]
                    if coords.shape[0] == px * py:
                        U_patch = model(coords).squeeze(-1).reshape(len(ix_vec), len(iy_vec))
                        
                        for a, ix in enumerate(ix_vec):
                            for b, iy in enumerate(iy_vec):
                                S[ix, iy] += U_patch[a, b]
                                W[ix, iy] += 1.0

        W = torch.where(W > 0, W, torch.ones_like(W))
        U_pred = S / W
        return Xg, Yg, U_pred

    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg, U_pred = self._predict_grid_via_patches(model, nx, ny)
        with torch.no_grad():
            U_true = self.u_star(Xg, Yg)
            num = torch.linalg.norm((U_pred - U_true).reshape(-1))
            den = torch.linalg.norm(U_true.reshape(-1))
            return (num / den).item()

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