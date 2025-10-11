# pinnlab/experiments/helmholtz2d_patch.py
"""
Time-dependent 2D Helmholtz (wave-type) with patch-based sampling.
PDE: u_tt - c^2 (u_xx + u_yy) + λ u = f(x, y, t)

We synthesize an analytic solution u*(x,y,t) and derive f accordingly.
BC: Dirichlet u|_{∂Ω} = u*(x,y,t)
IC: u(x,y,t0) = u*(x,y,t0),  and optionally  u_t(x,y,t0) = ∂_t u*(x,y,t0)
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from pinnlab.experiments.base import BaseExperiment_Patch
from pinnlab.data.patches import extract_xy_patches, attach_time
from pinnlab.data.geometries import linspace_2d
from pinnlab.utils.plotting import save_plots_2d


def _leaf(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach().requires_grad_(True)


class Helmholtz2D_patch(BaseExperiment_Patch):
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
        self.pad_mode_s = cfg.get("pad", {}).get("space", "reflect")   # reflect|replicate|circular|zero|none
        self.pad_mode_t = cfg.get("pad", {}).get("time", "reflect")

        # PDE constants
        self.c = float(cfg.get("c", 1.0))
        self.lam = float(cfg.get("lambda", 0.0))

        # Analytic solution parameters u*(x,y,t) = sin(a1πx) sin(a2πy) cos(ω t + φ)
        self.a1 = float(cfg.get("a1", 1.0))
        self.a2 = float(cfg.get("a2", 1.0))
        self.omega = float(cfg.get("omega", 2.0))
        self.phi = float(cfg.get("phi", 0.0))

        # Initial condition loss weighting for velocity term
        self.ic_v_weight = float(cfg.get("ic_v_weight", 1.0))

        # Normalization flags (optional small stabilizers)
        self.clip_time_grad = float(cfg.get("clip_time_grad", 0.0))  # 0 => off

        # Cached spacings (useful for evaluation grids)
        self.dx = (self.xb - self.xa) / max(1, self.gx - 1)
        self.dy = (self.yb - self.ya) / max(1, self.gy - 1)

        # Report
        print(
            f"[Helmholtz2D_patch] domain=([{self.xa},{self.xb}]×[{self.ya},{self.yb}]×[{self.t0},{self.t1}]), "
            f"patch={self.px}x{self.py}x{self.pt}, grid={self.gx}x{self.gy}x{self.gt}, "
            f"stride=({self.sx},{self.sy},{self.st}), c={self.c}, λ={self.lam}, "
            f"a=({self.a1},{self.a2}), ω={self.omega}, φ={self.phi}"
        )

    # -------- Analytic solution & forcing --------
    def u_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y) * torch.cos(self.omega * t + self.phi)

    def ut_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # ∂_t u* = -ω sin(ax) sin(ay) sin(ωt + φ)
        return -self.omega * torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y) * torch.sin(self.omega * t + self.phi)

    def f(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        From u_tt - c^2 ∇^2 u + λ u = f.
        For u* above, u_tt = -ω^2 u*, ∇^2 u* = -(a1^2+a2^2)π^2 u*.
        => f = (-ω^2 + c^2 (a1^2+a2^2) π^2 + λ) u*
        """
        coeff = (-self.omega ** 2) + (self.c ** 2) * (self.a1 ** 2 + self.a2 ** 2) * (math.pi ** 2) + self.lam
        return coeff * self.u_star(x, y, t)

    # -------- Sampling --------
    def sample_patches(self) -> Dict[str, torch.Tensor]:
        """
        Build spatiotemporal patches and masks in one pass.
        Returns dict with keys: 'coords','valid','is_bnd','is_ic','meta'
        - coords: [L, P, 3] where P = px*py*pt (after time attach)
        - valid : [L, P]   (0 for padded points, 1 valid)
        - is_bnd: [L, P]   (spatial boundary points across all times)
        - is_ic : [L, P]   (points at t == t0)
        """
        dev = self.device

        # Spatial sliding/unfold
        sp = extract_xy_patches(
            xa=self.xa, xb=self.xb, ya=self.ya, yb=self.yb,
            nx=self.gx, ny=self.gy,
            kx=self.px, ky=self.py,
            sx=self.sx, sy=self.sy,
            pad_mode=self.pad_mode_s,
            device=dev,
        )
        # Attach temporal windows (length pt, stride st)
        st_p = attach_time(
            sp,
            t0=self.t0, t1=self.t1, nt=self.gt,
            kt=self.pt, st=self.st,
            pad_mode_t=self.pad_mode_t,
        )
        return st_p  # coords:[L, P3, 3], valid:[L,P3], is_bnd:[L,P3], is_ic:[L,P3]

    def sample_batch(self, *_args, **_kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Train loop expects these keys to decide which losses to compute.
        We'll reuse the same coords/masks for all three heads.
        """
        P = self.sample_patches()
        return {
            "X_f": P,              # for PDE residual (interior)
            "X_b": {"coords": P["coords"], "mask": P["is_bnd"]},  # for BC
            "X_0": {"coords": P["coords"], "mask": P["is_ic"]},   # for IC
        }

    # -------- Losses --------
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

    def pde_residual_loss(self, model, batch) -> torch.Tensor:
        """
        r = u_tt - c^2 (u_xx + u_yy) + λ u - f = 0  on valid interior points (not on spatial boundary).
        """
        dev = self.device
        P = batch["X_f"]
        coords = P["coords"].to(dev)      # [L, P, 3]
        valid = P["valid"].to(dev) > 0.5  # [L, P]
        is_bnd = P["is_bnd"].to(dev) > 0.5

        # Keep valid non-boundary points
        keep = (valid & (~is_bnd)).reshape(-1)
        if not torch.any(keep):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        Xall = coords.reshape(-1, 3)
        X = _leaf(Xall[keep])             # [N,3] (x,y,t)

        # Forward
        U = self._forward_points(model, X)        # [N,1]

        # First derivatives
        gU = torch.autograd.grad(U, X, torch.ones_like(U), create_graph=True, retain_graph=True)[0]  # [N,3]
        u_x = gU[:, 0:1]
        u_y = gU[:, 1:2]
        u_t = gU[:, 2:3]

        # Second derivatives
        gux = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        guy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
        gut = torch.autograd.grad(u_t, X, torch.ones_like(u_t), create_graph=True, retain_graph=True)[0]
        u_xx = gux[:, 0:1]
        u_yy = guy[:, 1:2]
        u_tt = gut[:, 2:1+2]  # [:,2:3]

        # PDE residual
        fxyt = self.f(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        r = u_tt - (self.c ** 2) * (u_xx + u_yy) + self.lam * U - fxyt

        loss = (r ** 2).mean()
        return loss

    def boundary_loss(self, model, batch) -> torch.Tensor:
        """
        Dirichlet: u(x,y,t) = u*(x,y,t) for spatial boundary over all times.
        """
        dev = self.device
        if batch.get("X_b") is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        coords = batch["X_b"]["coords"].to(dev)      # [L,P,3]
        bmask  = batch["X_b"]["mask"].to(dev) > 0.5  # [L,P]
        keep = bmask.reshape(-1)
        if not torch.any(keep):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        X = coords.reshape(-1, 3)[keep]              # [Nb,3]
        U = self._forward_points(model, _leaf(X))    # [Nb,1]
        U_ref = self.u_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        return ((U - U_ref) ** 2).mean()

    def initial_loss(self, model, batch) -> torch.Tensor:
        """
        IC: u(x,y,t0) = u*(x,y,t0) and (optional) u_t(x,y,t0) = ∂_t u*(x,y,t0).
        Controlled by self.ic_v_weight.
        """
        dev = self.device
        if batch.get("X_0") is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        coords = batch["X_0"]["coords"].to(dev)     # [L,P,3]
        imask  = batch["X_0"]["mask"].to(dev) > 0.5 # [L,P]
        keep = imask.reshape(-1)
        if not torch.any(keep):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        X = coords.reshape(-1, 3)[keep]             # [N0,3], t≈t0
        X = _leaf(X)

        # Displacement IC
        U = self._forward_points(model, X)          # [N0,1]
        U_ref = self.u_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        L_u = ((U - U_ref) ** 2).mean()

        # Velocity IC (optional)
        L_v = torch.tensor(0.0, device=dev)
        if self.ic_v_weight > 0.0:
            gU = torch.autograd.grad(U, X, torch.ones_like(U), create_graph=True, retain_graph=True)[0]
            u_t = gU[:, 2:3]
            ut_ref = self.ut_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
            L_v = ((u_t - ut_ref) ** 2).mean()

        return L_u + self.ic_v_weight * L_v

    # -------- Evaluation & Plots --------
    def relative_l2_on_grid(self, model, grid_cfg) -> float:
        """
        Evaluate mean relative L2 across a few time slices (nt_eval).
        grid_cfg expects: nx, ny, nt (nt optional; default=5).
        """
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
                coords = torch.cat([XY, T], dim=1)                     # [nx*ny, 3]
                up = self._forward_points(model, coords).reshape(nx, ny)
                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                num = torch.linalg.norm((up - ut).reshape(-1))
                den = torch.linalg.norm(ut.reshape(-1)) + 1e-12
                rels.append((num / den).item())
            return float(sum(rels) / max(1, len(rels)))

    def plot_final(self, model, grid_cfg, out_dir):
        """
        Save 2D heatmaps (true/pred/abs_error) at a few time slices.
        """
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
                up = self._forward_points(model, coords).reshape(nx, ny)
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
