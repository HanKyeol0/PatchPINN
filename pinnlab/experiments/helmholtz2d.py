# pinnlab/experiments/helmholtz2d.py
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

class Helmholtz2D(BaseExperiment):
    """
    PDE: u_tt - c^2 (u_xx + u_yy) + λ u = f(x,y,t)
    Analytic target (for supervision of BC/IC/metrics):
        u* = sin(a1πx) sin(a2πy) cos(ω t + φ)
        f  = (-ω^2 + c^2 (a1^2+a2^2) π^2 + λ) u*
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
        self.pad_mode_s = cfg.get("pad", {}).get("xy", "none")
        self.pad_mode_t = cfg.get("pad", {}).get("t", "none")

        # PDE constants
        self.c   = float(cfg.get("c", 1.0))
        self.lam = float(cfg.get("lambda", 0.0))

        # Analytic u* params
        self.a1   = float(cfg.get("a1", 1.0))
        self.a2   = float(cfg.get("a2", 1.0))
        self.omega= float(cfg.get("omega", 2.0))
        self.phi  = float(cfg.get("phi", 0.0))

        # Boundary condition
        self.bc_type = cfg.get("bc_type", "dirichlet")  # "dirichlet" or "analytic" ("periodic" not implemented yet)
        self.bc_value = cfg.get("bc_value", 0.0)  # used only if bc_type=="dirichlet"

        # IC velocity weight (0.0 disables velocity IC)
        self.ic_v_weight = float(cfg.get("ic_v_weight", 1.0))

        # Grid spacings for FD
        self.dx = ((self.xb - self.xa) / (self.gx-1)) * 0.1
        self.dy = ((self.yb - self.ya) / (self.gy-1)) * 0.1
        self.dt = ((self.t1 - self.t0) / (self.gt-1)) * 0.1

        print(
            f"dx={self.dx:.6g}, dy={self.dy:.6g}, dt={self.dt:.6g}; "
            f"patch={self.px}x{self.py}x{self.pt}, stride=({self.sx},{self.sy},{self.st})"
        )

    # -------- Analytic solution & forcing --------
    def u_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y) * torch.cos(self.omega * t + self.phi)

    def ut_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -self.omega * torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y) * torch.sin(self.omega * t + self.phi)

    def f(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        coeff = (-self.omega ** 2) + (self.c ** 2) * (self.a1 ** 2 + self.a2 ** 2) * (math.pi ** 2) + self.lam
        return coeff * self.u_star(x, y, t)

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
        # sp["coords"]: [L, px*py, 2] (left bottom -> right top)
        
        st_p = attach_time(
            sp,
            t0=self.t0, t1=self.t1, nt=self.gt,
            kt=self.pt, st=self.st,
            pad_mode_t=self.pad_mode_t,
        )
        return st_p

    def prepare_epoch_patch_bank(self):
        """
        Build (or rebuild) the epoch's patch bank once.
        Cheaper than forward; main cost is just allocating coords.
        """
        # You already have this sampler pipeline; reuse it:
        self._patch_bank = self.sample_patches()      # dict with coords[N], valid, is_bnd, is_ic

    def sample_minibatch(self, k_patches: int, shuffle: bool = True):
        """
        Return a sliced view of the current patch bank with only k_patches windows.
        Assumes self._patch_bank['coords'] is [B, P, 3]-like (batch of patches).
        """
        assert hasattr(self, "_patch_bank") and self._patch_bank is not None, \
               "Call prepare_epoch_patch_bank() at start of epoch."
        P = self._patch_bank
        B = P["coords"].shape[0]
        if shuffle:
            idx = torch.randperm(B, device=self.device)[:min(k_patches, B)]
        else:
            idx = torch.arange(0, min(k_patches, B), device=self.device)
        def _slice(d):
            out = {}
            for k, v in d.items():
                if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == B:
                    out[k] = v.index_select(0, idx)
                else:
                    out[k] = v
            return out
        return {
            "X_f": _slice(P),
            "X_b": {"coords": P["coords"].index_select(0, idx),
                    "mask":   P["is_bnd"].index_select(0, idx)},
            "X_0": {"coords": P["coords"].index_select(0, idx),
                    "mask":   P["is_ic"].index_select(0, idx)},
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
    
    def _safe_interior_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Mark centers that are at least 1 grid step away from each global boundary
        so that central differences (±dx, ±dy, ±dt) stay in-domain.
        coords: [B, N, 3]
        """
        x, y, t = coords[..., 0], coords[..., 1], coords[..., 2]
        # integer indices on the global grid
        ix = torch.round((x - self.xa) / self.dx)
        iy = torch.round((y - self.ya) / self.dy)
        it = torch.round((t - self.t0) / self.dt)
        # need indices in [1, G-2] for central stencil
        mask = (
            (ix >= 1) & (ix <= (self.gx - 2)) &
            (iy >= 1) & (iy <= (self.gy - 2)) &
            (it >= 1) & (it <= (self.gt - 2))
        )
        return mask
    
    def _pde_residual_via_queries(self, model, P: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Central FD using on-the-fly model queries at ±dx, ±dy, ±dt around each retained center.
        Works even when px/py/pt < 3 (no interior cells exist inside a patch).
        """
        dev = self.device
        C = P["coords"].to(dev)   # [B, px*py*pt, 3]
        # valid  = P["valid"].to(dev)    # [B, px*py*pt]
        # is_bnd = P["is_bnd"].to(dev)   # [B, px*py*pt]

        if C.numel() == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # neighbor offsets
        ex = torch.tensor([self.dx, 0.0, 0.0], device=dev).view(1, 1, 3)
        ey = torch.tensor([0.0, self.dy, 0.0], device=dev).view(1, 1, 3)
        et = torch.tensor([0.0, 0.0, self.dt], device=dev).view(1, 1, 3)
        
        neigh = [
            C,                   # 0: center
            C + ex, C - ex,      # 1: +x, 2: -x
            C + ey, C - ey,      # 3: +y, 4: -y
            C + et, C - et,      # 5: +t, 6: -t
        ]
        
        results = []
        for patch in neigh:
            result = model(patch)
            results.append(result)
            
        Uc = results[0].squeeze(2)
        Uxp, Uxm = results[1].squeeze(2), results[2].squeeze(2)
        Uyp, Uym = results[3].squeeze(2), results[4].squeeze(2)
        Utp, Utm = results[5].squeeze(2), results[6].squeeze(2)

        # Central second derivatives
        Uxx = (Uxp - 2*Uc + Uxm) / (self.dx * self.dx)
        Uyy = (Uyp - 2*Uc + Uym) / (self.dy * self.dy)
        Utt = (Utp - 2*Uc + Utm) / (self.dt * self.dt)

        # Forcing at centers
        fx = self.f(C[..., 0], C[..., 1], C[..., 2])

        # Residual: u_tt - c^2 (u_xx + u_yy) + λ u - f = 0
        R = Utt - (self.c ** 2) * (Uxx + Uyy) + self.lam * Uc - fx

        return (R ** 2).mean()

    # ======== Losses ========
    def pde_residual_loss(self, model, batch) -> torch.Tensor:
        # Always use query-halo FD; robust for any patch size
        return self._pde_residual_via_queries(model, batch["X_f"])

    def boundary_loss(self, model, batch) -> torch.Tensor:
        dev = self.device
        if batch.get("X_b") is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        C = batch["X_b"]["coords"].to(dev)      # [B, px*py*pt, 3]
        bmask  = batch["X_b"]["mask"].to(dev) > 0.5  # [B, px*py*pt]
        keep = bmask.reshape(-1)
        if not torch.any(keep):
            return torch.tensor(0.0, device=dev, requires_grad=True)
        
        U = model(C) # ; U = U.view(-1, 1) if U.dim() == 1 else U[..., :1]
        U = U.squeeze(2).reshape(-1)[keep]
        
        if self.bc_type == "dirichlet":
            U_ref = torch.full_like(U, self.bc_value, device=dev, dtype=U.dtype)
        elif self.bc_type == "analytic":
            U_ref = self.u_star(C[..., 0], C[..., 1], C[..., 2])
            U_ref = U_ref.reshape(-1)[keep]
        return ((U - U_ref) ** 2).mean()

    def initial_loss(self, model, batch) -> torch.Tensor:
        """
        IC on displacement at t≈t0 using the provided mask (is_ic),
        plus optional velocity IC using forward FD in time (t and t+dt).
        """
        dev = self.device
        P0 = batch["X_0"]
        if P0 is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        C = P0["coords"].to(dev)      # [B, px*py*pt, 3]
        imask  = (P0["mask"].to(dev) > 0.5)  # [B, px*py*pt]
        imask = imask.reshape(-1)

        if C.numel() == 0 or not torch.any(imask):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # Displacement IC: u ≈ u_star at t≈t0 positions
        U_pred = model(C) # [B, px*py*pt, 1]
        U_pred = U_pred.squeeze(2).reshape(-1)[imask] # remain only IC points
        U_ref  = self.u_star(C[...,0], C[...,1], C[...,2]) # [B, px*py*pt]
        U_ref  = U_ref.reshape(-1)[imask] # remain only IC points
        L_u = ((U_pred - U_ref) ** 2).mean()

        # # Optional velocity IC
        # if self.ic_v_weight <= 0.0:
        #     return L_u

        # # Keep only those IC points whose t + dt is still inside [t0, t1]
        # it = torch.round((C[:,2] - self.t0) / self.dt)
        # ok_next = (it <= (self.gt - 2))   # has valid next step
        # if not torch.any(ok_next):
        #     return L_u

        # B_next = C[ok_next]
        # B_next = torch.stack([B_next[:,0], B_next[:,1], B_next[:,2] + self.dt], dim=1)

        # U_t    = U_pred[ok_next]                      # at t
        # U_tp1  = model(B_next)
        # if U_tp1.dim() == 1: U_tp1 = U_tp1[:, None]
        # U_tp1  = U_tp1[:, :1]

        # # forward difference ut ≈ (U(t+dt) - U(t)) / dt
        # U_t_fd = (U_tp1 - U_t) / self.dt

        # ut_ref = self.ut_star(B_next[:,0], B_next[:,1], (B_next[:,2] - self.dt)) \
        #            .unsqueeze(-1)  # reference at base time t
        # L_v = ((U_t_fd - ut_ref) ** 2).mean()

        return L_u # + self.ic_v_weight * L_v

    # -------- Evaluation & Plots (unchanged) --------
    def relative_l2_on_grid(self, model, grid_cfg) -> float:
        """
        Evaluate relative L2 error by tiling the (x,y,t) grid with non-overlapping
        patches of size (px,py,pt). If the grid size is not divisible by the patch
        size, we add a final patch that *overlaps* the previous one so that the
        right/top (and last time window) edges are covered.

        Patch traversal order:
        x: left -> right, y: bottom -> top, t: t0 -> t1.

        Combination rule:
        We write each prediction into a preallocated tensor initialized with NaNs.
        If a cell is already filled (can happen for the edge-overlap patches),
        we keep the first value and skip later writes to that cell.
        """
        dev = self.device
        nx, ny = int(grid_cfg["nx"]), int(grid_cfg["ny"])
        nt_eval = int(grid_cfg.get("nt", self.gt))  # fall back to train-time gt if not given

        # Build evaluation grid
        x = torch.linspace(self.xa, self.xb, nx, device=dev)
        y = torch.linspace(self.ya, self.yb, ny, device=dev)
        t = torch.linspace(self.t0, self.t1, nt_eval, device=dev)
        Xg, Yg = torch.meshgrid(x, y, indexing="xy")
        
        # Helpers to produce start indices with “edge-clamping” to include the last tile
        def _starts(n: int, k: int):
            # non-overlapping starts
            s = list(range(0, max(n - k + 1, 1), k))
            # ensure we cover the far edge; append last start (may overlap) if needed
            last = n - k
            if len(s) == 0 or s[-1] != last:
                s.append(last)
            return s

        kx, ky, kt = self.px, self.py, self.pt
        xs = _starts(nx, kx)
        ys = _starts(ny, ky)
        ts_starts = _starts(nt_eval, kt)

        # Preallocate prediction tensor (T, nx, ny) and a written-mask
        Up = torch.full((nt_eval, nx, ny), float("nan"), device=dev)
        written = torch.zeros((nt_eval, nx, ny), dtype=torch.bool, device=dev)
 
        with torch.no_grad():
            # Sweep time → y → x (t0..t1, bottom..top, left..right)
            for it0 in ts_starts:
                it1 = it0 + kt
                # time slice values and index range
                t_slice = t[it0:it1]                       # [kt]

                for jy0 in ys:
                    jy1 = jy0 + ky
                    y_vec = y[jy0:jy1]  # [ky]
                    
                    for ix0 in xs:
                        ix1 = ix0 + kx
                        x_vec = x[ix0:ix1]  # [kx]

                        # Extract the spatial block
                        X3D = x_vec.view(-1, 1, 1).repeat(1, ky, kt)
                        Y3D = y_vec.view(1, -1, 1).repeat(kx, 1, kt)
                        T3D = t_slice.view(1, 1, -1).repeat(kx, ky, 1)
                        
                        coords = torch.stack([X3D, Y3D, T3D], dim=-1).view(-1, 3)  # [kx*ky*kt, 3]

                        # Model prediction for this patch
                        up_patch = self._forward_points(model, coords)  # [kx*ky*kt, 1]
                        up_patch = up_patch[..., :1].squeeze(-1).view(kx, ky, kt)  # [kx,ky,kt]

                        # Scatter into the global tensor, skipping already-written cells
                        # We iterate in the same order to preserve the “first writer wins” rule.
                        for dt in range(kt):
                            tt = it0 + dt
                            # Build a mask of cells not yet written in this block
                            block_written = written[tt, ix0:ix1, jy0:jy1]
                            to_write = ~block_written
                            if to_write.any():
                                # Write only where not written yet
                                Up[tt, ix0:ix1, jy0:jy1][to_write] = up_patch[dt][to_write]
                                written[tt, ix0:ix1, jy0:jy1][to_write] = True

        # Ground truth over the *entire* domain for all times
        Utrue = torch.empty_like(Up)
        for it_idx in range(nt_eval):
            # broadcast t[it_idx] to grid shape
            tt = t[it_idx].expand_as(Xg)
            Utrue[it_idx] = self.u_star(Xg, Yg, tt)
        # Utrue shape: [nt_eval, nx, ny]

        valid = ~torch.isnan(Up)
        if not valid.any():
            return float("nan")

        num = torch.linalg.norm((Up[valid] - Utrue[valid]).reshape(-1))
        den = torch.linalg.norm(Utrue[valid].reshape(-1)) + 1e-12
        return float((num / den).item())

    def _tile_pred_true_on_grid(self, model, grid_cfg, nt_override: int = None):
        """
        Assemble predictions (and ground-truth) over the whole (x,y,t) domain
        using the same tiling logic and ordering as relative_l2_on_grid():
        - tiles: non-overlapping stride (px,py,pt), with edge-overlap if needed
        - ordering inside a tile: t-fast, then y, then x
        - combination rule: first-writer wins
        Returns:
        x (nx,), y (ny,), ts (T,), U_pred_T (T,nx,ny), U_true_T (T,nx,ny)
        """
        dev = self.device
        nx, ny = int(grid_cfg['nx']), int(grid_cfg['ny'])
        nt_eval = int(grid_cfg.get('nt', self.gt))
        if nt_override is not None:
            nt_eval = int(nt_override)

        # 1) axes and base grids
        x = torch.linspace(self.xa, self.xb, nx, device=dev)
        y = torch.linspace(self.ya, self.yb, ny, device=dev)
        ts = torch.linspace(self.t0, self.t1, nt_eval, device=dev)
        Xg, Yg = torch.meshgrid(x, y, indexing="xy")   # [nx,ny]

        # 2) start indices (non-overlap + edge-clamp)
        def _starts(n: int, k: int):
            s = list(range(0, max(n - k + 1, 1), k))
            last = n - k
            if len(s) == 0 or s[-1] != last:
                s.append(last)
            return s

        kx, ky, kt = self.px, self.py, self.pt
        xs = _starts(nx, kx)
        ys = _starts(ny, ky)
        ts_starts = _starts(nt_eval, kt)

        # 3) buffers
        U_pred_T = torch.full((nt_eval, nx, ny), float("nan"), device=dev)
        written  = torch.zeros((nt_eval, nx, ny), dtype=torch.bool, device=dev)

        with torch.no_grad():
            for it0 in ts_starts:
                it1 = it0 + kt
                t_slice = ts[it0:it1]            # [kt]

                for jy0 in ys:
                    jy1 = jy0 + ky
                    y_vec = y[jy0:jy1]          # [ky]

                    for ix0 in xs:
                        ix1 = ix0 + kx
                        x_vec = x[ix0:ix1]      # [kx]

                        # Build coords with t-fast → y → x
                        X3D = x_vec.view(-1, 1, 1).repeat(1, ky, kt)     # [kx,ky,kt]
                        Y3D = y_vec.view(1, -1, 1).repeat(kx, 1, kt)     # [kx,ky,kt]
                        T3D = t_slice.view(1, 1, -1).repeat(kx, ky, 1)   # [kx,ky,kt]
                        coords = torch.stack([X3D, Y3D, T3D], dim=-1).reshape(-1, 3)  # [kx*ky*kt,3]

                        # Forward
                        up = self._forward_points(model, coords)         # [P] or [P, C]
                        if up.dim() == 1:
                            up = up.unsqueeze(-1)
                        up = up[..., :1].squeeze(-1).reshape(kx, ky, kt) # [kx,ky,kt]

                        # Scatter: first-writer wins
                        for dt in range(kt):
                            tt = it0 + dt
                            block_written = written[tt, ix0:ix1, jy0:jy1]    # [kx,ky]
                            to_write = ~block_written
                            if to_write.any():
                                plane = up[:, :, dt]                          # [kx,ky]
                                U_pred_T[tt, ix0:ix1, jy0:jy1][to_write] = plane[to_write]
                                written[tt, ix0:ix1, jy0:jy1][to_write] = True

        # 4) ground-truth
        U_true_T = torch.empty_like(U_pred_T)
        for it in range(nt_eval):
            tval = ts[it].expand_as(Xg)
            U_true_T[it] = self.u_star(Xg, Yg, tval)

        return x, y, ts, U_pred_T, U_true_T

    def plot_final(self, model, grid_cfg, out_dir):
        """
        Save 2D triptych plots (true / pred / |err|) at a few evenly spaced times.
        Uses tiling with t→y→x order and first-writer-wins, consistent with training.
        Returns a dict of figure-name → path.
        """
        os.makedirs(out_dir, exist_ok=True)
        x, y, ts, U_pred_T, U_true_T = self._tile_pred_true_on_grid(model, grid_cfg)

        # pick 3~4 time indices
        T = ts.numel()
        picks = [0, T // 2, T - 1] if T >= 3 else list(range(T))
        paths = {}

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for it in picks:
            prefix = f"t{it:03d}_t{float(ts[it]):.4f}"
            u_true = U_true_T[it].detach().cpu().numpy()   # [nx,ny]
            u_pred = U_pred_T[it].detach().cpu().numpy()   # [nx,ny]
            figs = save_plots_2d(x_np, y_np, u_true, u_pred, out_dir, prefix)
            paths.update(figs)
        return paths

    def make_video(self, model, grid_cfg, out_dir, nt_video=None, fps=10, filename="evolution.mp4"):
        import numpy as np  # <-- add this import (inside or at file top)

        os.makedirs(out_dir, exist_ok=True)
        x, y, ts, U_pred_T, U_true_T = self._tile_pred_true_on_grid(model, grid_cfg, nt_override=nt_video)

        # --- robust global color scales using NumPy (handles NaNs on all torch versions) ---
        stack_np = torch.stack([U_true_T, U_pred_T], dim=0).detach().cpu().numpy()  # [2, T, nx, ny]
        Umin = float(np.nanmin(stack_np))
        Umax = float(np.nanmax(stack_np))

        err_np = torch.abs(U_true_T - U_pred_T).detach().cpu().numpy()
        if np.isfinite(err_np).any():
            err_vmax = float(np.nanpercentile(err_np, 99.5))  # robust cap for error colormap
        else:
            err_vmax = None
        # -----------------------------------------------------------------------------------

        # To numpy
        x_np   = x.detach().cpu().numpy()
        y_np   = y.detach().cpu().numpy()
        ts_np  = ts.detach().cpu().numpy()
        Utr_np = U_true_T.detach().cpu().numpy()
        Upr_np = U_pred_T.detach().cpu().numpy()

        out_path = os.path.join(out_dir, filename)
        return save_video_2d(
            x_np, y_np, Utr_np, Upr_np, ts_np,
            out_path=out_path, fps=fps,
            vmin=Umin, vmax=Umax, err_vmax=err_vmax, prefix="frame",
        )
