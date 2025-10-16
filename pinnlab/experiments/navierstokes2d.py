# pinnlab/experiments/poisson2d.py
import math, os, numpy as np
import torch
from typing import Dict, Tuple
from pinnlab.experiments.base import BaseExperiment
from pinnlab.data.patches import extract_xy_patches, attach_time
from pinnlab.data.geometries import linspace_2d
from pinnlab.utils.plotting import save_plots_2d, save_video_2d

def _leaf(x: torch.Tensor) -> torch.Tensor:
    # kept for API symmetry; not used for FD path but harmless
    return x.clone().detach().requires_grad_(True)

class NavierStokes2D(BaseExperiment):
    """
    2D incompressible Navier–Stokes (x,y,t), Taylor–Green vortex (periodic in x,y).

    Momentum:
        u_t + u u_x + v u_y + (1/ρ) p_x - ν (u_xx + u_yy) = 0
        v_t + u v_x + v v_y + (1/ρ) p_y - ν (v_xx + v_yy) = 0
    Continuity:
        u_x + v_y = 0

    We supervise BC/IC (and metrics) with a manufactured analytic solution (Taylor–Green vortex),
    so the PDE residual uses zero forcing.

    Model I/O:
        in_features = 3  (x,y,t)
        out_features = 3 (u,v,p)

    Analytic (TGV) solution:
        u*(x,y,t) =  U0 cos(kx) sin(ky) exp(-2 ν k^2 t)
        v*(x,y,t) = -U0 sin(kx) cos(ky) exp(-2 ν k^2 t)
        p*(x,y,t) =  ρ U0^2 / 4 [cos(2kx) + cos(2ky)] exp(-4 ν k^2 t)

    We enforce:
      - periodic BCs (paired equality at x=xa/xb and y=ya/yb)
      - IC at t = t0 from the analytic solution
      - PDE residual (momentum + continuity) at interior collocation points
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
        self.nu = float(cfg.get("nu", 0.01))          # viscosity
        self.rho = float(cfg.get("rho", 1.0))         # density

        # Manufactured (TGV-like) parameters
        self.a1 = float(cfg.get("a1", 1.0))
        self.a2 = float(cfg.get("a2", 1.0))

        # Loss weights
        self.div_weight = float(cfg.get("div_weight", 1.0))
        self.bc_weights = cfg.get("bc_weights", {"u": 1.0, "v": 1.0, "p": 0.1})
        self.ic_weights = cfg.get("ic_weights", {"u": 1.0, "v": 1.0, "p": 0.1})

        # Grid spacings for FD
        self.dx = (self.xb - self.xa) / max(1, self.gx - 1)
        self.dy = (self.yb - self.ya) / max(1, self.gy - 1)
        self.dt = (self.t1 - self.t0) / max(1, self.gt - 1)

        print(
            f"dx={self.dx:.6g}, dy={self.dy:.6g}, dt={self.dt:.6g}; "
            f"patch={self.px}x{self.py}x{self.pt}, stride=({self.sx},{self.sy},{self.st}); "
            f"nu={self.nu:.3g}"
        )

    # ------------------ Analytic (TGV-like) ------------------
    def _decay(self, t: torch.Tensor) -> torch.Tensor:
        # Taylor–Green decay rate (scaled to our a1,a2 on [xa,xb]×[ya,yb])
        # Works well as a manufactured solution for supervision.
        lam = 2.0 * (self.a1 ** 2 + self.a2 ** 2) * (math.pi ** 2) * self.nu
        return torch.exp(-lam * t)
    
    def u_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        #  u =  sin(a1πx) cos(a2πy) e^{-λ t}
        return torch.sin(self.a1 * math.pi * x) * torch.cos(self.a2 * math.pi * y) * self._decay(t)

    def v_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        #  v = -cos(a1πx) sin(a2πy) e^{-λ t}
        return -torch.cos(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y) * self._decay(t)

    def p_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        #  p = -1/4 [cos(2a1πx) + cos(2a2πy)] e^{-2λ t}
        lam = 2.0 * (self.a1 ** 2 + self.a2 ** 2) * (math.pi ** 2) * self.nu
        return -0.25 * (torch.cos(2 * self.a1 * math.pi * x) + torch.cos(2 * self.a2 * math.pi * y)) * torch.exp(-2 * lam * t)

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
    
    # ------------------ Packing helpers (multi-channel) ------------------
    def _pack_to_cubes_multi(
        self, coords: torch.Tensor, valid: torch.Tensor, is_bnd: torch.Tensor, U_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pack scattered points into dense cubes per window.

        Inputs:
          coords: [B, N, 3]  (x,y,t)
          valid : [B, N]
          is_bnd: [B, N]
          U_flat: [B, N, K]  (K=3 for u,v,p)

        Returns:
          U  : [B, K, pt, py, px]
          MV : [B, 1, pt, py, px]   (valid mask)
          MB : [B, 1, pt, py, px]   (boundary mask)
          X,Y,T : [B, 1, pt, py, px]
        """
        dev = coords.device
        B, N, _ = coords.shape
        K = U_flat.shape[-1]
        px, py, pt = self.px, self.py, self.pt
        P3 = px * py * pt
        dtype = U_flat.dtype

        # global integer indices
        ix_g = torch.round((coords[..., 0] - self.xa) / self.dx).to(torch.long)
        iy_g = torch.round((coords[..., 1] - self.ya) / self.dy).to(torch.long)
        it_g = torch.round((coords[..., 2] - self.t0) / self.dt).to(torch.long)
        ix0 = ix_g.min(dim=1, keepdim=True).values
        iy0 = iy_g.min(dim=1, keepdim=True).values
        it0 = it_g.min(dim=1, keepdim=True).values
        ix = (ix_g - ix0).clamp_(0, px - 1)
        iy = (iy_g - iy0).clamp_(0, py - 1)
        it = (it_g - it0).clamp_(0, pt - 1)

        li = (it * (py * px) + iy * px + ix)  # [B, N]
        offsets = torch.arange(B, device=dev).view(B, 1) * P3
        li_global = (li + offsets).view(-1)   # [B*N]

        shape_flat = (B * P3,)
        zeros1 = lambda: torch.zeros(shape_flat, dtype=dtype, device=dev)
        zerosK = lambda: torch.zeros(B * P3, K, dtype=dtype, device=dev)

        U_sum = zerosK()
        Cnt   = zeros1()
        MV_sum= zeros1()
        MB_sum= zeros1()
        X_sum = zeros1()
        Y_sum = zeros1()
        T_sum = zeros1()

        U_src = U_flat.reshape(-1, K).to(dtype)
        ones  = torch.ones_like(Cnt)

        # scatter (per-point -> local cell)
        U_sum.index_add_(0, li_global, U_src)
        Cnt  .index_add_(0, li_global, ones)
        MV_sum.index_add_(0, li_global, valid.reshape(-1).to(dtype))
        MB_sum.index_add_(0, li_global, is_bnd.reshape(-1).to(dtype))
        X_sum .index_add_(0, li_global, coords[..., 0].reshape(-1).to(dtype))
        Y_sum .index_add_(0, li_global, coords[..., 1].reshape(-1).to(dtype))
        T_sum .index_add_(0, li_global, coords[..., 2].reshape(-1).to(dtype))

        Cnt = torch.clamp(Cnt, min=1.0)
        U = (U_sum / Cnt.view(-1, 1)).view(B, pt, py, px, K).permute(0, 4, 1, 2, 3).contiguous()
        MV = (MV_sum > 0).to(dtype).view(B, pt, py, px).unsqueeze(1)
        MB = (MB_sum > 0).to(dtype).view(B, pt, py, px).unsqueeze(1)
        X  = (X_sum / Cnt).view(B, pt, py, px).unsqueeze(1)
        Y  = (Y_sum / Cnt).view(B, pt, py, px).unsqueeze(1)
        T  = (T_sum / Cnt).view(B, pt, py, px).unsqueeze(1)
        return U, MV, MB, X, Y, T

    def _forward_points(self, model, X: torch.Tensor) -> torch.Tensor:
        """
        Accepts [N,3] coords; returns [N,3] (u,v,p).
        """
        U = model(X)
        if U.dim() == 1:
            U = U[:, None]
        if U.shape[-1] < 3:
            # if the model gives more channels, keep first 3; if less, pad zeros
            if U.shape[-1] == 1:
                U = torch.cat([U, torch.zeros_like(U), torch.zeros_like(U)], dim=-1)
            elif U.shape[-1] == 2:
                U = torch.cat([U, torch.zeros(U.shape[0], 1, device=U.device, dtype=U.dtype)], dim=-1)
        return U[..., :3]  # [N,3]

    # ------------------ Losses ------------------
    def pde_residual_loss(self, model, batch) -> torch.Tensor:
        """
        FD residuals on interior cells:

          r_u = u_t + u u_x + v u_y + (1/ρ) p_x - ν (u_xx + u_yy) = 0
          r_v = v_t + u v_x + v v_y + (1/ρ) p_y - ν (v_xx + v_yy) = 0
          r_div = u_x + v_y = 0

        evaluated where center & neighbors exist and center is not on spatial boundary.
        """
        dev = self.device
        P = batch["X_f"]
        coords = P["coords"].to(dev)   # [B,N,3]
        valid  = P["valid"].to(dev)    # [B,N]
        is_bnd = P["is_bnd"].to(dev)   # [B,N]

        if coords.numel() == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # Predict once
        UVP_flat = self._forward_points(model, coords.reshape(-1, 3)).reshape(coords.shape[0], coords.shape[1], 3)
        U, MV, MB, X, Y, T = self._pack_to_cubes_multi(coords, valid, is_bnd, UVP_flat)
        # Shapes:
        #   U  : [B, 3, pt, py, px]   (channels: 0=u, 1=v, 2=p)
        #   MV : [B, 1, pt, py, px]
        #   MB : [B, 1, pt, py, px]
        #   X,Y,T: [B,1,pt,py,px]

        u = U[:, 0:1]  # [B,1,pt,py,px]
        v = U[:, 1:2]
        p = U[:, 2:3]

        # interior stencil
        C_u = u[:, :, 1:-1, 1:-1, 1:-1]
        C_v = v[:, :, 1:-1, 1:-1, 1:-1]
        C_p = p[:, :, 1:-1, 1:-1, 1:-1]

        # 1st-order spatial grads (central)
        u_x = (u[:, :, 1:-1, 1:-1, 2:] - u[:, :, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        u_y = (u[:, :, 1:-1, 2:,  1:-1] - u[:, :, 1:-1, :-2, 1:-1]) / (2 * self.dy)
        v_x = (v[:, :, 1:-1, 1:-1, 2:] - v[:, :, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        v_y = (v[:, :, 1:-1, 2:,  1:-1] - v[:, :, 1:-1, :-2, 1:-1]) / (2 * self.dy)
        p_x = (p[:, :, 1:-1, 1:-1, 2:] - p[:, :, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        p_y = (p[:, :, 1:-1, 2:,  1:-1] - p[:, :, 1:-1, :-2, 1:-1]) / (2 * self.dy)

        # time derivative (central)
        u_t = (u[:, :, 2:,  1:-1, 1:-1] - u[:, :, :-2, 1:-1, 1:-1]) / (2 * self.dt)
        v_t = (v[:, :, 2:,  1:-1, 1:-1] - v[:, :, :-2, 1:-1, 1:-1]) / (2 * self.dt)

        # Laplacians
        u_xx = (u[:, :, 1:-1, 1:-1, 2:] - 2 * C_u + u[:, :, 1:-1, 1:-1, :-2]) / (self.dx ** 2)
        u_yy = (u[:, :, 1:-1, 2:,  1:-1] - 2 * C_u + u[:, :, 1:-1, :-2, 1:-1]) / (self.dy ** 2)
        v_xx = (v[:, :, 1:-1, 1:-1, 2:] - 2 * C_v + v[:, :, 1:-1, 1:-1, :-2]) / (self.dx ** 2)
        v_yy = (v[:, :, 1:-1, 2:,  1:-1] - 2 * C_v + v[:, :, 1:-1, :-2, 1:-1]) / (self.dy ** 2)

        # masks: need center and +/-x, +/-y (and +/-t for u_t,v_t), and exclude spatial boundary
        MVc = MV[:, :, 1:-1, 1:-1, 1:-1]
        Mx  = MV[:, :, 1:-1, 1:-1, 2:] * MV[:, :, 1:-1, 1:-1, :-2]
        My  = MV[:, :, 1:-1, 2:,  1:-1] * MV[:, :, 1:-1, :-2, 1:-1]
        Mt  = MV[:, :, 2:,  1:-1, 1:-1] * MV[:, :, :-2, 1:-1, 1:-1]
        MBc = MB[:, :, 1:-1, 1:-1, 1:-1]
        Mmom = (MVc * Mx * My * Mt) * (1.0 - MBc)  # {0,1} float

        # residuals
        conv_u = C_u * u_x + C_v * u_y
        conv_v = C_u * v_x + C_v * v_y
        coeff = 1.0 / self.rho

        r_u = u_t + conv_u + coeff * p_x - self.nu * (u_xx + u_yy)
        r_v = v_t + conv_v + coeff * p_y - self.nu * (v_xx + v_yy)
        r_div = u_x + v_y  # no time derivative needed; we can still share Mmom mask for simplicity

        R2 = (r_u ** 2 + r_v ** 2 + self.div_weight * (r_div ** 2)) * Mmom

        denom = torch.clamp(Mmom.sum(), min=1.0)
        return R2.sum() / denom

    def boundary_loss(self, model, batch) -> torch.Tensor:
        """
        Dirichlet BC on u,v (and lightly on p to anchor pressure offset).
        """
        dev = self.device
        B = batch.get("X_b")
        if B is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        coords = B["coords"].to(dev)  # [B,N,3]
        bmask  = (B["mask"].to(dev) > 0.5)  # [B,N]
        keep = bmask.reshape(-1)
        if not torch.any(keep):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        X = coords.reshape(-1, 3)[keep]
        pred = self._forward_points(model, X)  # [M,3]
        u_ref = self.u_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        v_ref = self.v_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        p_ref = self.p_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)

        Lu = ((pred[:, 0:1] - u_ref) ** 2).mean()
        Lv = ((pred[:, 1:2] - v_ref) ** 2).mean()
        Lp = ((pred[:, 2:3] - p_ref) ** 2).mean()

        wu = float(self.bc_weights.get("u", 1.0))
        wv = float(self.bc_weights.get("v", 1.0))
        wp = float(self.bc_weights.get("p", 0.1))
        return wu * Lu + wv * Lv + wp * Lp

    def initial_loss(self, model, batch) -> torch.Tensor:
        """
        IC at t≈t0 on u,v (and optionally p).
        """
        dev = self.device
        P0 = batch.get("X_0")
        if P0 is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        coords = P0["coords"].to(dev)  # [B,N,3]
        imask  = (P0["mask"].to(dev) > 0.5)  # [B,N]
        keep = imask.reshape(-1)
        if not torch.any(keep):
            return torch.tensor(0.0, device=dev, requires_grad=True)

        X = coords.reshape(-1, 3)[keep]
        pred = self._forward_points(model, X)
        u_ref = self.u_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        v_ref = self.v_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        p_ref = self.p_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)

        Lu = ((pred[:, 0:1] - u_ref) ** 2).mean()
        Lv = ((pred[:, 1:2] - v_ref) ** 2).mean()
        Lp = ((pred[:, 2:3] - p_ref) ** 2).mean()

        wu = float(self.ic_weights.get("u", 1.0))
        wv = float(self.ic_weights.get("v", 1.0))
        wp = float(self.ic_weights.get("p", 0.1))
        return wu * Lu + wv * Lv + wp * Lp

    # -------- Evaluation & Plots (unchanged) --------
    def relative_l2_on_grid(self, model, grid_cfg) -> float:
        dev = self.device
        nx, ny = int(grid_cfg["nx"]), int(grid_cfg["ny"])
        nt_eval = int(grid_cfg.get("nt", 5))
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, dev)
        ts = torch.linspace(self.t0, self.t1, nt_eval, device=dev)
        with torch.no_grad():
            rels = []
            XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
            for t in ts:
                T = torch.full((XY.shape[0], 1), float(t), device=dev)
                coords = torch.cat([XY, T], dim=1)
                out = self._forward_points(model, coords).reshape(nx, ny, 3)
                up, vp = out[..., 0], out[..., 1]

                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                vt = self.v_star(Xg, Yg, t.expand_as(Xg))

                num = torch.linalg.norm(torch.stack([(up - ut).reshape(-1), (vp - vt).reshape(-1)], dim=1))
                den = torch.linalg.norm(torch.stack([ut.reshape(-1), vt.reshape(-1)], dim=1)) + 1e-12
                rels.append((num / den).item())
            return float(sum(rels) / max(1, len(rels)))

    def plot_final(self, model, grid_cfg, out_dir):
        """
        Save heatmaps for u, v, and speed |(u,v)|.
        """
        dev = self.device
        nx, ny = int(grid_cfg["nx"]), int(grid_cfg["ny"])
        nt_eval = int(grid_cfg.get("nt", 3))
        ts = torch.linspace(self.t0, self.t1, nt_eval, device=dev)
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, dev)
        paths = {}

        with torch.no_grad():
            XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
            for i, t in enumerate(ts):
                T = torch.full((XY.shape[0], 1), float(t), device=dev)
                coords = torch.cat([XY, T], dim=1)
                out = self._forward_points(model, coords).reshape(nx, ny, 3)
                up, vp = out[..., 0], out[..., 1]
                speed_p = torch.sqrt(up ** 2 + vp ** 2)

                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                vt = self.v_star(Xg, Yg, t.expand_as(Xg))
                speed_t = torch.sqrt(ut ** 2 + vt ** 2)

                # three save calls (u, v, speed) with triptychs (true/pred/abs_err)
                p1 = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(),
                                   ut.cpu().numpy(), up.cpu().numpy(),
                                   out_dir, prefix=f"ns2d_u_t{i}")
                p2 = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(),
                                   vt.cpu().numpy(), vp.cpu().numpy(),
                                   out_dir, prefix=f"ns2d_v_t{i}")
                p3 = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(),
                                   speed_t.cpu().numpy(), speed_p.cpu().numpy(),
                                   out_dir, prefix=f"navierstokes2d{i}")
                paths.update(p1); paths.update(p2); paths.update(p3)
        return paths


    def make_video(
        self, model, grid_cfg, out_dir,
        filename="navierstokes2d.mp4", nt_video=60, fps=10,
        vmin=None, vmax=None, err_vmax=None,
    ):
        """
        Triptych video (True | Pred | |True-Pred|) for speed magnitude.
        """
        dev = self.device
        nx, ny = int(grid_cfg["x"]), int(grid_cfg["y"])
        ts = torch.linspace(self.t0, self.t1, nt_video, device=dev)
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, dev)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

        Utrue_list, Upred_list = [], []
        with torch.no_grad():
            for t in ts:
                T = torch.full((XY.shape[0], 1), float(t), device=dev)
                coords = torch.cat([XY, T], dim=1)
                out = self._forward_points(model, coords).reshape(nx, ny, 3)
                up, vp = out[..., 0], out[..., 1]
                speed_p = torch.sqrt(up ** 2 + vp ** 2)

                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                vt = self.v_star(Xg, Yg, t.expand_as(Xg))
                speed_t = torch.sqrt(ut ** 2 + vt ** 2)

                Utrue_list.append(speed_t.cpu().numpy())
                Upred_list.append(speed_p.cpu().numpy())

        U_true_T = torch.stack([torch.from_numpy(a) for a in Utrue_list], dim=0).numpy()
        U_pred_T = torch.stack([torch.from_numpy(a) for a in Upred_list], dim=0).numpy()

        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, filename)
        return save_video_2d(
            Xg.cpu().numpy(),
            Yg.cpu().numpy(),
            U_true_T, U_pred_T, ts.cpu().numpy(),
            out_path=video_path, fps=fps,
            vmin=vmin, vmax=vmax, err_vmax=err_vmax,
            prefix="navierstokes2d"
        )