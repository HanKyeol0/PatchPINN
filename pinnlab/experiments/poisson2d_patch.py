# pinnlab/experiments/poisson2d_patch.py
import torch
from pinnlab.experiments.poisson2d import Poisson2D
from pinnlab.experiments.base import make_leaf, grad_sum
from pinnlab.data.patches import extract_xy_patches, attach_time

class Poisson2D_Patch(Poisson2D):
    """
    Patch-based variant of Poisson2D (steady or time-dependent).
    exp_cfg['grid'] : {nx, ny[, nt]}
    exp_cfg['patch']: {kx, ky, sx, sy, pad_mode_xy: 'none'|'zero'|'reflect'|'replicate'|'circular',
                       # for time-dependent:
                       kt: 1|3|..., st: 1|..., pad_mode_t: 'none'|'zero'|'reflect'|'replicate'|'circular'}
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        g = cfg.get("grid", {})
        self.nx = int(g.get("nx", 65))
        self.ny = int(g.get("ny", 65))
        self.nt = int(g.get("nt", 17)) if self.time_dep else None

        p = cfg.get("patch", {})
        self.kx = int(p.get("kx", 9));   self.ky = int(p.get("ky", 9))
        self.sx = int(p.get("sx", 4));   self.sy = int(p.get("sy", 4))
        self.pad_mode_xy = str(p.get("pad_mode_xy", "zero"))

        if self.time_dep:
            self.kt = int(p.get("kt", 3));    self.st = int(p.get("st", 1))
            self.pad_mode_t = str(p.get("pad_mode_t", "zero"))

    # ---------- batching ----------
    def sample_batch(self, n_f, n_b, n_0):
        # We ignore (n_f, n_b, n_0); patch count is determined by grid/stride
        xa, xb = self.rect.xa, self.rect.xb; ya, yb = self.rect.ya, self.rect.yb
        core = extract_xy_patches(
            xa, xb, ya, yb,
            self.nx, self.ny,
            self.kx, self.ky, self.sx, self.sy,
            self.pad_mode_xy, self.rect.device
        )
        if self.time_dep:
            batch = attach_time(
                core, self.t0, self.t1, self.nt,
                self.kt, self.st, self.pad_mode_t
            )
        else:
            # promote to 3D-like API with t dropped
            batch = {**core, "is_ic": torch.zeros_like(core["valid"])}

        return {
            "patch_coords": batch["coords"],  # [B,P,D]
            "valid": batch["valid"],
            "is_bnd": batch["is_bnd"],
            "is_ic": batch["is_ic"],
            "X_f": torch.empty(0, device=self.device)  # <- sentinel for legacy guard
        }

    # ---------- losses (patch-aware) ----------
    def pde_residual_loss(self, model, batch):
        C = batch["patch_coords"]                  # [B,P,D]
        V = batch["valid"]                         # [B,P]
        X = make_leaf(C)                           # keep shape [B,P,D] as a leaf for AD
        u = model.forward_patches(X, V)            # [B,P,1]
        du = grad_sum(u, X)                         # [B,P,D]

        if self.time_dep:
            u_x, u_y, u_t = du[:,0:1], du[:,1:2], du[:,2:3]
            u_xx = grad_sum(u_x, X)[:,0:1]
            u_yy = grad_sum(u_y, X)[:,1:2]
            x, y, t = X[:,0:1], X[:,1:2], X[:,2:3]
            res = u_t - self.kappa * (u_xx + u_yy) - self.f_time(x, y, t)
        else:
            u_x, u_y = du[:,0:1], du[:,1:2]
            u_xx = grad_sum(u_x, X)[:,0:1]
            u_yy = grad_sum(u_y, X)[:,1:2]
            x, y = X[:,0:1], X[:,1:2]
            res = -(u_xx + u_yy) - self.f_steady(x, y)

        res = res.reshape(C.shape[0], C.shape[1])     # [B,P]
        return (res.pow(2) * V).reshape(-1)           # only valid points contribute

    def boundary_loss(self, model, batch):
        if "is_bnd" not in batch: 
            return torch.tensor(0.0, device=self.device)
        C = batch["patch_coords"]; M = (batch["is_bnd"] > 0.5) & (batch["valid"] > 0.5)
        if M.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        Xb = C[M].reshape(-1, C.shape[-1])

        if self.time_dep:
            ub = self.u_star_time(Xb[:,0:1], Xb[:,1:2], Xb[:,2:3])
        else:
            ub = self.u_star_steady(Xb[:,0:1], Xb[:,1:2])

        pred = model(Xb)
        return (pred - ub).pow(2)

    def initial_loss(self, model, batch):
        if not self.time_dep: 
            return torch.tensor(0.0, device=self.device)
        C = batch["patch_coords"]; M = (batch["is_ic"] > 0.5) & (batch["valid"] > 0.5)
        if M.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        X0 = C[M].reshape(-1, C.shape[-1])
        u0 = self.u_star_time(X0[:,0:1], X0[:,1:2], X0[:,2:3])
        pred = model(X0)
        return (pred - u0).pow(2)