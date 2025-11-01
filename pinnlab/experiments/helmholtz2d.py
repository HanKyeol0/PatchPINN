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
        self.pad_mode_s = cfg.get("pad", {}).get("space", "reflect")
        self.pad_mode_t = cfg.get("pad", {}).get("time", "reflect")

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
        self.dx = (self.xb - self.xa) / max(1, self.gx - 1)
        self.dy = (self.yb - self.ya) / max(1, self.gy - 1)
        self.dt = (self.t1 - self.t0) / max(1, self.gt - 1)

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
        P = self.sample_patches()      # dict with coords[N], valid, is_bnd, is_ic
        self._patch_bank = P           # store for this epoch

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
    
    def _pde_residual_via_queries(self, model, P: Dict[str, torch.Tensor], ep) -> torch.Tensor:
        """
        Central FD using on-the-fly model queries at ±dx, ±dy, ±dt around each retained center.
        Works even when px/py/pt < 3 (no interior cells exist inside a patch).
        """
        dev = self.device
        coords = P["coords"].to(dev)   # [B, px*py*pt, 3]
        valid  = P["valid"].to(dev)    # [B, px*py*pt]
        is_bnd = P["is_bnd"].to(dev)   # [B, px*py*pt]

        if coords.numel() == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # keep only interior (not domain boundary, has neighbors one step away)
        keep_mask = (valid > 0.5) & (~(is_bnd > 0.5)) & self._safe_interior_mask(coords) # [B, px*py*pt]
        
        if ep==0:
            print("keep mask")
            print(keep_mask)
        if not torch.any(keep_mask):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        C = coords[keep_mask]          # [M, 3]
        if ep==0:
            print("C shape")
            print(C.shape)
        M = C.shape[0]
        if M == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # neighbor offsets
        ex = torch.tensor([self.dx, 0.0, 0.0], device=dev).view(1, 1, 3)
        ey = torch.tensor([0.0, self.dy, 0.0], device=dev).view(1, 1, 3)
        et = torch.tensor([0.0, 0.0, self.dt], device=dev).view(1, 1, 3)

        C_ = C.view(M, 1, 3)
        if ep==0:
            print("Sampled PDE residual points (first minibatch of epoch):")
            print(C_)
            print("C_ len")
            print(C_.shape)
        neigh = torch.cat([
            C_,                     # 0: center
            C_ + ex, C_ - ex,      # 1: +x, 2: -x
            C_ + ey, C_ - ey,      # 3: +y, 4: -y
            C_ + et, C_ - et,      # 5: +t, 6: -t
        ], dim=1).reshape(-1, 3)   # [M*7, 3]

        # One forward pass for all needed points (no AD on inputs)
        U_all = model(neigh, ep)                   # [M*7, 1] or [M*7]
        if U_all.dim() == 1:
            U_all = U_all[:, None]
        U_all = U_all[:, :1]                   # ensure scalar field
        U_all = U_all.view(M, 7, 1)            # [M, 7, 1]

        Uc   = U_all[:, 0, 0]                  # center
        Uxp  = U_all[:, 1, 0]; Uxm = U_all[:, 2, 0]
        Uyp  = U_all[:, 3, 0]; Uym = U_all[:, 4, 0]
        Utp  = U_all[:, 5, 0]; Utm = U_all[:, 6, 0]

        # Central second derivatives
        Uxx = (Uxp - 2*Uc + Uxm) / (self.dx * self.dx)
        Uyy = (Uyp - 2*Uc + Uym) / (self.dy * self.dy)
        Utt = (Utp - 2*Uc + Utm) / (self.dt * self.dt)

        # Forcing at centers
        fx = self.f(C[:, 0], C[:, 1], C[:, 2])

        # Residual: u_tt - c^2 (u_xx + u_yy) + λ u - f = 0
        R = Utt - (self.c ** 2) * (Uxx + Uyy) + self.lam * Uc - fx

        return (R ** 2).mean()

    # ======== Losses ========
    def pde_residual_loss(self, model, batch, ep) -> torch.Tensor:
        # Always use query-halo FD; robust for any patch size
        return self._pde_residual_via_queries(model, batch["X_f"], ep=ep)

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
        
        if self.bc_type == "dirichlet":
            U_ref = torch.full_like(U, self.bc_value, device=dev, dtype=U.dtype)
        elif self.bc_type == "analytic":
            U_ref = self.u_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
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

        coords = P0["coords"].to(dev)      # [B, N, 3]
        imask  = (P0["mask"].to(dev) > 0.5)  # [B, N]

        if coords.numel() == 0 or not torch.any(imask):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # Select IC locations
        C_ic = coords[imask].view(-1, 3)     # [M,3]
        if C_ic.shape[0] == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # Displacement IC: u ≈ u_star at t≈t0 positions
        U_pred = model(C_ic)
        if U_pred.dim() == 1: U_pred = U_pred[:, None]
        U_pred = U_pred[:, :1]                         # [M,1]
        U_ref  = self.u_star(C_ic[:,0], C_ic[:,1], C_ic[:,2]).unsqueeze(-1)
        L_u = ((U_pred - U_ref) ** 2).mean()

        # Optional velocity IC
        if self.ic_v_weight <= 0.0:
            return L_u

        # Keep only those IC points whose t + dt is still inside [t0, t1]
        it = torch.round((C_ic[:,2] - self.t0) / self.dt)
        ok_next = (it <= (self.gt - 2))   # has valid next step
        if not torch.any(ok_next):
            return L_u

        B_next = C_ic[ok_next]
        B_next = torch.stack([B_next[:,0], B_next[:,1], B_next[:,2] + self.dt], dim=1)

        U_t    = U_pred[ok_next]                      # at t
        U_tp1  = model(B_next)
        if U_tp1.dim() == 1: U_tp1 = U_tp1[:, None]
        U_tp1  = U_tp1[:, :1]

        # forward difference ut ≈ (U(t+dt) - U(t)) / dt
        U_t_fd = (U_tp1 - U_t) / self.dt

        ut_ref = self.ut_star(B_next[:,0], B_next[:,1], (B_next[:,2] - self.dt)) \
                   .unsqueeze(-1)  # reference at base time t
        L_v = ((U_t_fd - ut_ref) ** 2).mean()

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
                    prefix=f"helmholtz2d_time_t{i}"
                )
                paths.update(figs)
        return paths


    def make_video(
        self, model, grid_cfg, out_dir,
        filename="helmholtz2d_time.mp4", nt_video=60, fps=10,
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
            prefix="helmholtz2d"
        )