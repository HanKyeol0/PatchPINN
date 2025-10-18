# pinnlab/experiments/burgers2d.py
import math, os, numpy as np
import torch
from typing import Dict, Tuple
from pinnlab.experiments.base import BaseExperiment
from pinnlab.data.patches import extract_xy_patches, attach_time
from pinnlab.data.geometries import linspace_2d
from pinnlab.utils.plotting import save_plots_2d, save_video_2d

class Burgers2D(BaseExperiment):
    """
    2D viscous Burgers (x,y,t) with manufactured forcing:

        u_t + u u_x + v u_y = ν (u_xx + u_yy) + f_u(x,y,t)
        v_t + u v_x + v v_y = ν (v_xx + v_yy) + f_v(x,y,t)

    We supervise BC/IC using an analytic manufactured solution (TGV-like),
    and define f_u, f_v so that (u*, v*) satisfy the PDE exactly.

    Model I/O:
        in_features = 3  (x,y,t)
        out_features = 2 (u,v)

    Manufactured solution (decaying sin/cos):
        Let ax = a1 * π * x,  by = a2 * π * y,  D(t) = exp(-λ t)
        u*(x,y,t) =  sin(ax) cos(by) D(t)
        v*(x,y,t) = -cos(ax) sin(by) D(t)

        With λ free (config), forcing is:
          f_u = u*_t + u* u*_x + v* u*_y - ν (u*_{xx} + u*_{yy})
          f_v = v*_t + u* v*_x + v* v*_y - ν (v*_{xx} + v*_{yy})
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        # Domain
        self.xa, self.xb = cfg["domain"]["x"]
        self.ya, self.yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]

        # Patch/grid
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
        self.lam = float(cfg.get("lambda", 2.0))      # temporal decay of u*,v*

        # Manufactured params
        self.a1 = float(cfg.get("a1", 1.0))
        self.a2 = float(cfg.get("a2", 1.0))

        # (optional) weights
        self.bc_weights = cfg.get("bc_weights", {"u": 1.0, "v": 1.0})
        self.ic_weights = cfg.get("ic_weights", {"u": 1.0, "v": 1.0})

        # Grid spacings for FD
        self.dx = (self.xb - self.xa) / max(1, self.gx - 1)
        self.dy = (self.yb - self.ya) / max(1, self.gy - 1)
        self.dt = (self.t1 - self.t0) / max(1, self.gt - 1)

        print(
            f"dx={self.dx:.6g}, dy={self.dy:.6g}, dt={self.dt:.6g}; "
            f"patch={self.px}x{self.py}x{self.pt}, stride=({self.sx},{self.sy},{self.st}); "
            f"nu={self.nu:.3g}, lambda={self.lam:.3g}"
        )

    # ------------------ Analytic u*, v* & forcing f ------------------
    def _ax(self, x): return self.a1 * math.pi * x
    def _by(self, y): return self.a2 * math.pi * y
    def _D(self, t):  return torch.exp(-self.lam * t)

    def u_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(self._ax(x)) * torch.cos(self._by(y)) * self._D(t)

    def v_star(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -torch.cos(self._ax(x)) * torch.sin(self._by(y)) * self._D(t)

    def _derivs_u(self, x, y, t):
        ax, by, D = self._ax(x), self._by(y), self._D(t)
        sAx, cAx = torch.sin(ax), torch.cos(ax)
        sBy, cBy = torch.sin(by), torch.cos(by)
        ap, bp   = self.a1 * math.pi, self.a2 * math.pi

        u  = sAx * cBy * D
        ut = -self.lam * u
        ux = ap * cAx * cBy * D
        uy = -bp * sAx * sBy * D
        uxx = -(ap ** 2) * u
        uyy = -(bp ** 2) * u
        return u, ut, ux, uy, uxx, uyy

    def _derivs_v(self, x, y, t):
        ax, by, D = self._ax(x), self._by(y), self._D(t)
        sAx, cAx = torch.sin(ax), torch.cos(ax)
        sBy, cBy = torch.sin(by), torch.cos(by)
        ap, bp   = self.a1 * math.pi, self.a2 * math.pi

        v  = -cAx * sBy * D
        vt = -self.lam * v
        vx =  ap * sAx * sBy * D
        vy = -bp * cAx * cBy * D
        vxx = -(ap ** 2) * v
        vyy = -(bp ** 2) * v
        return v, vt, vx, vy, vxx, vyy

    def f(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute f_u, f_v so that (u*,v*) satisfy the Burgers equations.
        """
        u, ut, ux, uy, uxx, uyy = self._derivs_u(x, y, t)
        v, vt, vx, vy, vxx, vyy = self._derivs_v(x, y, t)
        fu = ut + u * ux + v * uy - self.nu * (uxx + uyy)
        fv = vt + u * vx + v * vy - self.nu * (vxx + vyy)
        return fu, fv

    # ------------------ Sampling ------------------
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
    ):
        """
        Pack scattered points into dense cubes per window.

        Inputs:
          coords: [B, N, 3]  (x,y,t)
          valid : [B, N]
          is_bnd: [B, N]
          U_flat: [B, N, 2]  (u,v)

        Returns:
          U  : [B, 2, pt, py, px]
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
        Accepts [N,3] coords; returns [N,2] (u,v).
        """
        U = model(X)
        if U.dim() == 1:
            U = U[:, None]
        if U.shape[-1] < 2:
            # If model has 1 channel, pad v=0 to avoid shape errors during early tests.
            if U.shape[-1] == 1:
                U = torch.cat([U, torch.zeros_like(U)], dim=-1)
        return U[..., :2]  # [N,2]

    # ------------------ Losses ------------------
    def pde_residual_loss(self, model, batch) -> torch.Tensor:
        """
        FD residuals on interior cells:

          r_u = u_t + u u_x + v u_y - ν (u_xx + u_yy) - f_u = 0
          r_v = v_t + u v_x + v v_y - ν (v_xx + v_yy) - f_v = 0
        """
        dev = self.device
        P = batch["X_f"]
        coords = P["coords"].to(dev)   # [B,N,3]
        valid  = P["valid"].to(dev)    # [B,N]
        is_bnd = P["is_bnd"].to(dev)   # [B,N]

        if coords.numel() == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # Predict once
        UV_flat = self._forward_points(model, coords.reshape(-1, 3)).reshape(coords.shape[0], coords.shape[1], 2)
        U, MV, MB, X, Y, T = self._pack_to_cubes_multi(coords, valid, is_bnd, UV_flat)
        # Shapes:
        #   U  : [B, 2, pt, py, px]  (0=u, 1=v)
        #   MV : [B, 1, pt, py, px]
        #   MB : [B, 1, pt, py, px]

        u = U[:, 0:1]  # [B,1,pt,py,px]
        v = U[:, 1:2]

        # interior stencil
        C_u = u[:, :, 1:-1, 1:-1, 1:-1]
        C_v = v[:, :, 1:-1, 1:-1, 1:-1]

        # central diffs (space/time)
        u_x = (u[:, :, 1:-1, 1:-1, 2:] - u[:, :, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        u_y = (u[:, :, 1:-1, 2:,  1:-1] - u[:, :, 1:-1, :-2, 1:-1]) / (2 * self.dy)
        v_x = (v[:, :, 1:-1, 1:-1, 2:] - v[:, :, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        v_y = (v[:, :, 1:-1, 2:,  1:-1] - v[:, :, 1:-1, :-2, 1:-1]) / (2 * self.dy)
        u_t = (u[:, :, 2:,  1:-1, 1:-1] - u[:, :, :-2, 1:-1, 1:-1]) / (2 * self.dt)
        v_t = (v[:, :, 2:,  1:-1, 1:-1] - v[:, :, :-2, 1:-1, 1:-1]) / (2 * self.dt)

        u_xx = (u[:, :, 1:-1, 1:-1, 2:] - 2 * C_u + u[:, :, 1:-1, 1:-1, :-2]) / (self.dx ** 2)
        u_yy = (u[:, :, 1:-1, 2:,  1:-1] - 2 * C_u + u[:, :, 1:-1, :-2, 1:-1]) / (self.dy ** 2)
        v_xx = (v[:, :, 1:-1, 1:-1, 2:] - 2 * C_v + v[:, :, 1:-1, 1:-1, :-2]) / (self.dx ** 2)
        v_yy = (v[:, :, 1:-1, 2:,  1:-1] - 2 * C_v + v[:, :, 1:-1, :-2, 1:-1]) / (self.dy ** 2)

        # masks: center & neighbors valid & exclude spatial boundary
        MVc = MV[:, :, 1:-1, 1:-1, 1:-1]
        Mx  = MV[:, :, 1:-1, 1:-1, 2:] * MV[:, :, 1:-1, 1:-1, :-2]
        My  = MV[:, :, 1:-1, 2:,  1:-1] * MV[:, :, 1:-1, :-2, 1:-1]
        Mt  = MV[:, :, 2:,  1:-1, 1:-1] * MV[:, :, :-2, 1:-1, 1:-1]
        MBc = MB[:, :, 1:-1, 1:-1, 1:-1]
        MOK = (MVc * Mx * My * Mt) * (1.0 - MBc)

        # forcing at centers
        Xc, Yc, Tc = X[:, :, 1:-1, 1:-1, 1:-1], Y[:, :, 1:-1, 1:-1, 1:-1], T[:, :, 1:-1, 1:-1, 1:-1]
        fu, fv = self.f(Xc.squeeze(1), Yc.squeeze(1), Tc.squeeze(1))
        fu = fu.unsqueeze(1); fv = fv.unsqueeze(1)  # [B,1,pt-2,py-2,px-2]

        # residuals
        r_u = u_t + C_u * u_x + C_v * u_y - self.nu * (u_xx + u_yy) - fu
        r_v = v_t + C_u * v_x + C_v * v_y - self.nu * (v_xx + v_yy) - fv
        R2 = (r_u ** 2 + r_v ** 2) * MOK

        denom = torch.clamp(MOK.sum(), min=1.0)
        return R2.sum() / denom

    def boundary_loss(self, model, batch) -> torch.Tensor:
        """
        Dirichlet BC on u,v from manufactured u*, v*.
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
        pred = self._forward_points(model, X)  # [M,2]
        u_ref = self.u_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)
        v_ref = self.v_star(X[:, 0], X[:, 1], X[:, 2]).unsqueeze(-1)

        Lu = ((pred[:, 0:1] - u_ref) ** 2).mean()
        Lv = ((pred[:, 1:2] - v_ref) ** 2).mean()

        wu = float(self.bc_weights.get("u", 1.0))
        wv = float(self.bc_weights.get("v", 1.0))
        return wu * Lu + wv * Lv

    def initial_loss(self, model, batch) -> torch.Tensor:
        """
        IC at t≈t0 on u,v from manufactured u*, v*.
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

        Lu = ((pred[:, 0:1] - u_ref) ** 2).mean()
        Lv = ((pred[:, 1:2] - v_ref) ** 2).mean()

        wu = float(self.ic_weights.get("u", 1.0))
        wv = float(self.ic_weights.get("v", 1.0))
        return wu * Lu + wv * Lv

    # ------------------ Evaluation & Plots ------------------
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
                out = self._forward_points(model, coords).reshape(nx, ny, 2)
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
                out = self._forward_points(model, coords).reshape(nx, ny, 2)
                up, vp = out[..., 0], out[..., 1]
                speed_p = torch.sqrt(up ** 2 + vp ** 2)

                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                vt = self.v_star(Xg, Yg, t.expand_as(Xg))
                speed_t = torch.sqrt(ut ** 2 + vt ** 2)

                p1 = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(),
                                   ut.cpu().numpy(), up.cpu().numpy(),
                                   out_dir, prefix=f"burgers2d_u_t{i}")
                p2 = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(),
                                   vt.cpu().numpy(), vp.cpu().numpy(),
                                   out_dir, prefix=f"burgers2d_v_t{i}")
                p3 = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(),
                                   speed_t.cpu().numpy(), speed_p.cpu().numpy(),
                                   out_dir, prefix=f"burgers2d{i}")
                paths.update(p1); paths.update(p2); paths.update(p3)
        return paths

    def make_video(
        self, model, grid_cfg, out_dir,
        filename="burgers2d.mp4", nt_video=60, fps=10,
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
                out = self._forward_points(model, coords).reshape(nx, ny, 2)
                up, vp = out[..., 0], out[..., 1]
                speed_p = torch.sqrt(up ** 2 + vp ** 2)

                ut = self.u_star(Xg, Yg, t.expand_as(Xg))
                vt = self.v_star(Xg, Yg, t.expand_as(Xg))
                speed_t = torch.sqrt(ut ** 2 + vt ** 2)

                Utrue_list.append(speed_t.cpu().numpy())
                Upred_list.append(speed_p.cpu().numpy())

        U_true_T = np.stack(Utrue_list, axis=0)
        U_pred_T = np.stack(Upred_list, axis=0)

        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, filename)
        return save_video_2d(
            Xg.cpu().numpy(),
            Yg.cpu().numpy(),
            U_true_T, U_pred_T, ts.cpu().numpy(),
            out_path=video_path, fps=fps,
            vmin=vmin, vmax=vmax, err_vmax=err_vmax,
            prefix="burgers2d"
        )
