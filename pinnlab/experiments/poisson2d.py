# pinnlab/experiments/poisson2d.py
import math, os, numpy as np
import torch
from typing import Dict
from pinnlab.experiments.base import BaseExperiment
from pinnlab.data.patches import extract_xy_patches, attach_time
from pinnlab.data.geometries import linspace_2d
from pinnlab.utils.plotting import save_plots_2d, save_video_2d

def _leaf(x: torch.Tensor) -> torch.Tensor:
    # kept for API symmetry; not used for FD path but harmless
    return x.clone().detach().requires_grad_(True)

class Poisson2D(BaseExperiment):
    """
    time-dependent Poisson:

    - Time-dependent Heat:
        u_t - κ ∇²u = f(x,y,t)
        Choose u*(x,y,t) = sin(pi x) sin(pi y) exp(-λ t)
        ∇²u* = -2π² u*,
        u_t = -λ u*
        residual = (-λ + 2 κ π²) u* - f
        If λ = 2 κ π²  =>  f ≡ 0 (homogeneous). BC and IC from u*.

    Config keys:
      kappa, lambda ('auto') if time_dependent
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        # Domain
        self.xa, self.xb = cfg["domain"]["x"]
        self.ya, self.yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]

        # Patch/grid (space & time)
        self.px = int(cfg["patch"]["x"])
        self.py = int(cfg["patch"]["y"])
        self.pt = int(cfg["patch"]["t"])
        self.gx = int(cfg["grid"]["x"])
        self.gy = int(cfg["grid"]["y"])
        self.gt = int(cfg["grid"]["t"])

        # Strides & padding
        self.sx = int(cfg.get("stride", {}).get("x", self.px))
        self.sy = int(cfg.get("stride", {}).get("y", self.py))
        self.st = int(cfg.get("stride", {}).get("t", self.pt))
        self.pad_mode_s = cfg.get("pad", {}).get("space", "reflect")
        self.pad_mode_t = cfg.get("pad", {}).get("time", "reflect")

        # PDE constants
        self.kappa = float(cfg.get("kappa", 1.0))
        self.lam   = float(cfg.get("lambda", 1.0))

        # IC velocity weight (0.0 disables velocity IC)
        self.ic_v_weight = float(cfg.get("ic_v_weight", 1.0))

        # Grid spacings for FD
        self.dx = (self.xb - self.xa) / max(1, self.gx - 1)
        self.dy = (self.yb - self.ya) / max(1, self.gy - 1)
        self.dt = (self.t1 - self.t0) / max(1, self.gt - 1)

        print(
            f"dx={self.dx:.6g}, dy={self.dy:.6g}, dt={self.dt:.6g}; "
            f"patch={self.px}x{self.py}x{self.pt}, stride=({self.sx},{self.sy},{self.st})"
        )

    # -------- Analytic solution & forcing --------
    def u_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.exp(-self.lam * t)

    def ut_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -self.lam * torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.exp(-self.lam * t)
    
    def del2_u_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -2 * (math.pi ** 2) * self.u_star(x, y, t)

    def f(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.ut_star(x, y, t) - self.kappa * self.del2_u_star(x, y, t)

    # -------- Sampling (same as before) --------
    def sample_patches(self) -> Dict[str, torch.Tensor]:
        sp = extract_xy_patches(
            xa=self.xa, xb=self.xb, ya=self.ya, yb=self.yb,
            nx=self.gx, ny=self.gy,
            kx=self.px, ky=self.py,
            sx=self.sx, sy=self.sy,
            pad_mode=self.pad_mode_s,
            device=self.device,
        )
        st_p = attach_time(
            sp,
            t0=self.t0, t1=self.t1, nt=self.gt,
            kt=self.pt, st=self.st,
            pad_mode_t=self.pad_mode_t,
        )
        return st_p

    def sample_batch(self, *_args, **_kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        P = self.sample_patches()
        return {
            "X_f": P,  # interior (use valid & ~is_bnd)
            "X_b": {"coords": P["coords"], "mask": P["is_bnd"]},
            "X_0": {"coords": P["coords"], "mask": P["is_ic"]},
        }

    # ======== Finite-difference helpers ========
    def _grid_indices_local(self, coords_s: torch.Tensor):
        """
        Convert (x,y,t) in a window to local integer indices (ix,iy,it)
        by snapping to global grid with dx,dy,dt, then shifting to local 0-based.
        """
        ix_g = torch.round((coords_s[:, 0] - self.xa) / self.dx).to(torch.long)
        iy_g = torch.round((coords_s[:, 1] - self.ya) / self.dy).to(torch.long)
        it_g = torch.round((coords_s[:, 2] - self.t0) / self.dt).to(torch.long)
        ix0, iy0, it0 = ix_g.min(), iy_g.min(), it_g.min()
        return ix_g - ix0, iy_g - iy0, it_g - it0

    def _fill_cube(self, shape, idx_tuple, values, dtype=None, device=None):
        """Create a dense [px,py,pt] cube and scatter values into it."""
        cube = torch.zeros(shape, dtype=values.dtype if dtype is None else dtype, device=values.device if device is None else device)
        cube[idx_tuple] = values
        return cube
    
    def _pack_to_cubes(self, coords, valid, is_bnd, U_flat):
        """
        Returns:
        U , MV, MB, X, Y, T  with dtype = U_flat.dtype  and shape [B, 1, pt, py, px]
        """
        dev = coords.device
        B, N, _ = coords.shape
        px, py, pt = self.px, self.py, self.pt
        P3 = px * py * pt
        dtype = U_flat.dtype  # <— use model output dtype consistently

        # integer indices
        ix_g = torch.round((coords[..., 0] - self.xa) / self.dx).to(torch.long)
        iy_g = torch.round((coords[..., 1] - self.ya) / self.dy).to(torch.long)
        it_g = torch.round((coords[..., 2] - self.t0) / self.dt).to(torch.long)
        ix0 = ix_g.min(dim=1, keepdim=True).values
        iy0 = iy_g.min(dim=1, keepdim=True).values
        it0 = it_g.min(dim=1, keepdim=True).values
        ix = (ix_g - ix0).clamp_(0, px - 1)
        iy = (iy_g - iy0).clamp_(0, py - 1)
        it = (it_g - it0).clamp_(0, pt - 1)

        li = (it * (py * px) + iy * px + ix)                   # [B,N]
        offsets = torch.arange(B, device=dev).view(B, 1) * P3
        li_global = (li + offsets).view(-1)                    # [B*N]

        shape_flat = (B * P3,)
        zeros = lambda: torch.zeros(shape_flat, device=dev, dtype=dtype)

        U_sum  = zeros(); Cnt   = zeros()
        MV_sum = zeros(); MB_sum= zeros()
        X_sum  = zeros(); Y_sum = zeros(); T_sum = zeros()

        # cast inputs to common dtype
        U_src   = U_flat.to(dtype).view(-1)
        MV_src  = valid.to(dtype).view(-1)
        MB_src  = is_bnd.to(dtype).view(-1)
        X_src   = coords[..., 0].to(dtype).view(-1)
        Y_src   = coords[..., 1].to(dtype).view(-1)
        T_src   = coords[..., 2].to(dtype).view(-1)
        one_src = torch.ones_like(U_src)

        U_sum.scatter_add_(0, li_global, U_src)
        Cnt.scatter_add_(0, li_global, one_src)
        MV_sum.scatter_add_(0, li_global, MV_src)
        MB_sum.scatter_add_(0, li_global, MB_src)
        X_sum.scatter_add_(0, li_global, X_src)
        Y_sum.scatter_add_(0, li_global, Y_src)
        T_sum.scatter_add_(0, li_global, T_src)

        Cnt = torch.clamp(Cnt, min=1.0)
        U  = (U_sum  / Cnt).view(B, pt, py, px).unsqueeze(1)
        MV = (MV_sum > 0).to(dtype).view(B, pt, py, px).unsqueeze(1)
        MB = (MB_sum > 0).to(dtype).view(B, pt, py, px).unsqueeze(1)
        X  = (X_sum  / Cnt).view(B, pt, py, px).unsqueeze(1)
        Y  = (Y_sum  / Cnt).view(B, pt, py, px).unsqueeze(1)
        T  = (T_sum  / Cnt).view(B, pt, py, px).unsqueeze(1)
        return U, MV, MB, X, Y, T

    def _forward_points(self, model, X: torch.Tensor) -> torch.Tensor:
        """
        model() can accept [N,3] (PatchFFN tolerates variable P).
        Returns [N,1]
        """
        U = model(X)
        if U.dim() == 1:
            U = U[:, None]
        if U.shape[-1] != 1:
            U = U[..., :1]
        return U  # [N,1]

    # ======== Losses ========
    def pde_residual_loss(self, model, batch) -> torch.Tensor:
        """
        r = u_t - κ ∇²u - f(x,y,t)
        """
        dev = self.device
        P = batch["X_f"]
        coords = P["coords"].to(dev)      # [B, N, 3] with B=L*Nt
        valid  = P["valid"].to(dev)       # [B, N]
        is_bnd = P["is_bnd"].to(dev)      # [B, N]

        if coords.numel() == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # Predict once for the whole batch
        U_pred = model(coords.reshape(-1, 3)).reshape(coords.shape[0], coords.shape[1])
        U, MV, MB, X, Y, T = self._pack_to_cubes(coords, valid, is_bnd, U_pred)

        # central differences on interior
        C   = U[:, :, 1:-1, 1:-1, 1:-1]
        Ut = (U[:, :, 2:,  1:-1, 1:-1] - U[:, :, :-2, 1:-1, 1:-1]) / (2.0 * self.dt)
        Uxx = (U[:, :, 1:-1, 1:-1, 2:]  - 2*C + U[:, :, 1:-1, 1:-1, :-2]) / (self.dx * self.dx)
        Uyy = (U[:, :, 1:-1, 2:,  1:-1] - 2*C + U[:, :, 1:-1, :-2, 1:-1]) / (self.dy * self.dy)
        Utt = (U[:, :, 2:,  1:-1, 1:-1] - 2*C + U[:, :, :-2, 1:-1, 1:-1]) / (self.dt * self.dt)

        Xc, Yc, Tc = X[:, :, 1:-1, 1:-1, 1:-1], Y[:, :, 1:-1, 1:-1, 1:-1], T[:, :, 1:-1, 1:-1, 1:-1]
        Fc = self.f(Xc.squeeze(1), Yc.squeeze(1), Tc.squeeze(1)).unsqueeze(1)

        # validity: center & 6 neighbors valid, and center is not spatial boundary
        MVc = MV[:, :, 1:-1, 1:-1, 1:-1]
        Mx  = MV[:, :, 1:-1, 1:-1, 2:] * MV[:, :, 1:-1, 1:-1, :-2]
        My  = MV[:, :, 1:-1, 2:,  1:-1] * MV[:, :, 1:-1, :-2, 1:-1]
        Mt  = MV[:, :, 2:,  1:-1, 1:-1] * MV[:, :, :-2, 1:-1, 1:-1]
        MBc = MB[:, :, 1:-1, 1:-1, 1:-1]
        MOK = (MVc * Mx * My * Mt) * (1.0 - MBc)  # float mask {0,1}

        R = Ut - self.kappa * (Uxx + Uyy) - Fc # u_t - κ ∇²u = f(x,y,t)
        R2 = (R * MOK) ** 2

        denom = torch.clamp(MOK.sum(), min=1.0)
        return R2.sum() / denom

    def boundary_loss(self, model, batch) -> torch.Tensor:
        dev = self.device
        if batch.get("X_b") is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        coords = batch["X_b"]["coords"].to(dev)      # [B,N,3]
        bmask  = batch["X_b"]["mask"].to(dev) > 0.5  # [B,N]
        keep = bmask.reshape(-1)
        if not torch.any(keep):
            return torch.tensor(0.0, device=dev, requires_grad=True)
        X = coords.reshape(-1, 3)[keep]
        U = model(X); U = U.view(-1, 1) if U.dim() == 1 else U[..., :1]
        U_ref = self.u_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        return ((U - U_ref) ** 2).mean()

    def initial_loss(self, model, batch) -> torch.Tensor:
        """
        Displacement IC at t≈t0, and optional velocity IC with FD (forward in time).
        No autograd on inputs.
        """
        dev = self.device
        P0 = batch["X_0"]
        if P0 is None:  # safety
            return torch.tensor(0.0, device=dev, requires_grad=True)

        coords = P0["coords"].to(dev)   # [B, N, 3]
        imask  = P0["mask"].to(dev)     # [B, N]  (is_ic mask from attach_time)

        if coords.numel() == 0 or not torch.any(imask > 0.5):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        U_pred = model(coords.reshape(-1, 3)).reshape(coords.shape[0], coords.shape[1])
        # valid=1 everywhere here (IC selection controls), boundary not needed for IC
        ones = torch.ones_like(imask)
        zeros= torch.zeros_like(imask)
        U, MV, MB, X, Y, T = self._pack_to_cubes(coords, ones, zeros, U_pred)

        # Displacement IC: positions marked is_ic
        # Pack IC mask to cube
        _, pt, py, px = U.shape[1], U.shape[2], U.shape[3], U.shape[4]
        B, N = imask.shape
        # reuse index mapping by recomputing local indices (same as in _pack_to_cubes)
        ix_g = torch.round((coords[..., 0] - self.xa) / self.dx).to(torch.long)
        iy_g = torch.round((coords[..., 1] - self.ya) / self.dy).to(torch.long)
        it_g = torch.round((coords[..., 2] - self.t0) / self.dt).to(torch.long)
        ix0 = ix_g.min(dim=1, keepdim=True).values; iy0 = iy_g.min(dim=1, keepdim=True).values; it0 = it_g.min(dim=1, keepdim=True).values
        ix = (ix_g - ix0).clamp_(0, self.px - 1); iy = (iy_g - iy0).clamp_(0, self.py - 1); it = (it_g - it0).clamp_(0, self.pt - 1)
        li = (it * (self.py * self.px) + iy * self.px + ix)
        offsets = torch.arange(B, device=dev).view(B, 1) * (self.pt * self.py * self.px)
        li_global = (li + offsets).view(-1)

        IC_flat = torch.zeros(B * self.pt * self.py * self.px, device=dev, dtype=coords.dtype)
        IC_flat.scatter_add_(0, li_global, imask.view(-1))
        IC = (IC_flat.view(B, self.pt, self.py, self.px).unsqueeze(1) > 0.5).to(coords.dtype)

        # Disp. IC loss
        Uref = self.u_star(X.squeeze(1), Y.squeeze(1), T.squeeze(1)).unsqueeze(1)
        # Uref = self.u_star(X, Y, T).unsqueeze(1) if X.dim()==4 else self.u_star(X.squeeze(1), Y.squeeze(1), T.squeeze(1)).unsqueeze(1)
        Lu_num = ((U - Uref) ** 2 * IC).sum()
        Lu_den = torch.clamp(IC.sum(), min=1.0)
        L_u = Lu_num / Lu_den

        # Velocity IC (forward difference at t0): optional
        if self.ic_v_weight <= 0.0 or self.pt < 2:
            return L_u

        # ut ≈ (U(t+dt) - U(t)) / dt  at IC locations that have a next-time neighbor
        Utf = (U[:, :, 1:, :, :] - U[:, :, :-1, :, :]) / self.dt
        ICf = IC[:, :, :-1, :, :]  # require base IC at t_k
        # also require the "next" time step is valid
        MV_next = MV[:, :, 1:, :, :]
        Mmask = ICf * MV_next
        # evaluate reference ut at the base time T[:, :, :-1]
        ut_ref = self.ut_star(X[:, :, :-1, :, :].squeeze(1), Y[:, :, :-1, :, :].squeeze(1), T[:, :, :-1, :, :].squeeze(1)).unsqueeze(1)

        Lv_num = ((Utf - ut_ref) ** 2 * Mmask).sum()
        Lv_den = torch.clamp(Mmask.sum(), min=1.0)
        L_v = Lv_num / Lv_den

        return L_u + self.ic_v_weight * L_v

    # -------- Evaluation & Plots (unchanged) --------
    def relative_l2_on_grid(self, model, grid_cfg) -> float:
        dev = self.device
        nx, ny = int(grid_cfg["nx"]), int(grid_cfg["ny"])
        nt_eval = int(grid_cfg.get("nt", 5))
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, dev)
        ts = torch.linspace(self.t0, self.t1, nt_eval, device=dev)
        with torch.no_grad():
            rels = []
            for t in ts:
                XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
                T = torch.full((XY.shape[0], 1), float(t), device=dev)
                coords = torch.cat([XY, T], dim=1)
                up = model(coords).reshape(nx, ny)
                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                num = torch.linalg.norm((up - ut).reshape(-1))
                den = torch.linalg.norm(ut.reshape(-1)) + 1e-12
                rels.append((num / den).item())
            return float(sum(rels) / max(1, len(rels)))

    def plot_final(self, model, grid_cfg, out_dir):
        dev = self.device
        nx, ny = int(grid_cfg["nx"]), int(grid_cfg["ny"])
        nt_eval = int(grid_cfg.get("nt", 3))
        ts = torch.linspace(self.t0, self.t1, nt_eval, device=dev)
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, dev)
        paths = {}
        with torch.no_grad():
            for i, t in enumerate(ts):
                XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
                T = torch.full((XY.shape[0], 1), float(t), device=dev)
                coords = torch.cat([XY, T], dim=1)
                up = model(coords).reshape(nx, ny)
                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                figs = save_plots_2d(
                    Xg.detach().cpu().numpy(),
                    Yg.detach().cpu().numpy(),
                    ut.detach().cpu().numpy(),
                    up.detach().cpu().numpy(),
                    out_dir,
                    prefix=f"poisson2d{i}"
                )
                paths.update(figs)
        return paths


    def make_video(
        self, model, grid_cfg, out_dir,
        filename="poisson2d.mp4", nt_video=60, fps=10,
        vmin=None, vmax=None, err_vmax=None,
    ):
        """
        Render a video of True | Pred | |True-Pred| across time.

        grid_cfg: {"nx": int, "ny": int}
        """

        dev = self.device
        nx, ny = int(grid_cfg["x"]), int(grid_cfg["y"])
        ts = torch.linspace(self.t0, self.t1, nt_video, device=dev)

        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, dev)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [nx*ny, 2]

        Utrue_list, Upred_list = [], []
        with torch.no_grad():
            for t in ts:
                T = torch.full((XY.shape[0], 1), float(t), device=dev)
                coords = torch.cat([XY, T], dim=1)                 # [nx*ny, 3]
                up = self._forward_points(model, coords).reshape(nx, ny)
                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                Utrue_list.append(ut.detach().cpu().numpy())
                Upred_list.append(up.detach().cpu().numpy())

        U_true_T = np.stack(Utrue_list, axis=0)  # [T, nx, ny]
        U_pred_T = np.stack(Upred_list, axis=0)  # [T, nx, ny]
        ts_np = ts.detach().cpu().numpy()

        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, filename)
        return save_video_2d(
            Xg.detach().cpu().numpy(),
            Yg.detach().cpu().numpy(),
            U_true_T, U_pred_T, ts_np,
            out_path=video_path, fps=fps,
            vmin=vmin, vmax=vmax, err_vmax=err_vmax,
            prefix="poisson2d"
        )