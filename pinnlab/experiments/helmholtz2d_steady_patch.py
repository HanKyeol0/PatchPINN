"""
helmholtz2d_steady_patch_fixed.py - Fixed version with proper gradient tracking
"""

import math
import numpy as np
import torch
from pinnlab.experiments.base_patch import BaseExperiment_Patch
from pinnlab.data.geometries import linspace_2d
from pinnlab.data.samplers import sample_patches_2d_steady
from pinnlab.utils.plotting import save_plots_2d


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
            device=self.device,
            stride_x=None,  # Will default to px (non-overlapping)
            stride_y=None,  # Will default to py (non-overlapping)
            boundary_fn=g_dirichlet,
        )
        x_f, x_b, u_b = out["interior_patches"], out["boundary_patches"], out["true_boundary"]
        return x_f, x_b, u_b
    
    def sample_batch(self, *_, **__):
        x_f, x_b, u_b = self.sample_patches()
        return {"X_f": x_f, "X_b": x_b, "u_b": u_b}
    
    def pde_residual_loss(self, model, batch):
        """
        FIXED: Compute PDE residual loss with proper gradient tracking.
        """
        device = self.device
        P = self.patch_size
        
        # Gather all patches (interior + boundary interior points)
        coords_list = []
        keep_mask_list = []
        
        # Pure interior patches
        for coords in batch.get("X_f", []):
            coords_list.append(coords)
            keep_mask_list.append(torch.ones(P, dtype=torch.bool, device=device))
        
        # Boundary patches (only interior points)
        for patch in batch.get("X_b", []):
            coords = patch["coords"]
            bmask = patch["boundary_mask"]
            coords_list.append(coords)
            keep_mask_list.append(~bmask)  # Keep only interior points
        
        if len(coords_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        errs = []
        bs = self.batch_size
        
        for start, coords_chunk in self._iterate_in_chunks(coords_list, bs):
            k_masks_chunk = keep_mask_list[start:start+len(coords_chunk)]
            
            # Stack coordinates
            C = torch.stack(coords_chunk, dim=0).to(device)  # [B,P,2]
            K = torch.stack(k_masks_chunk, dim=0).to(device)  # [B,P]
            B = C.size(0)
            
            # CRITICAL FIX: Make sure coordinates require gradients
            X = C.clone().detach().requires_grad_(True)  # [B,P,2]
            
            # Forward through model
            U = model(X)  # [B,P,1] or [B,P]
            if U.dim() == 2:
                U = U.unsqueeze(-1)  # Ensure [B,P,1]
            
            # Compute gradients using individual outputs
            # This ensures proper gradient computation
            u_x_list = []
            u_y_list = []
            u_xx_list = []
            u_yy_list = []
            
            for b in range(B):
                for p in range(P):
                    if not K[b, p]:  # Skip masked points
                        continue
                    
                    u_point = U[b, p, 0]
                    
                    # First derivatives
                    grad_u = torch.autograd.grad(
                        outputs=u_point,
                        inputs=X,
                        grad_outputs=torch.ones_like(u_point),
                        create_graph=True,
                        retain_graph=True
                    )[0]  # [B,P,2]
                    
                    u_x = grad_u[b, p, 0]
                    u_y = grad_u[b, p, 1]
                    
                    # Second derivatives
                    grad_ux = torch.autograd.grad(
                        outputs=u_x,
                        inputs=X,
                        grad_outputs=torch.ones_like(u_x),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    grad_uy = torch.autograd.grad(
                        outputs=u_y,
                        inputs=X,
                        grad_outputs=torch.ones_like(u_y),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    u_xx = grad_ux[b, p, 0]
                    u_yy = grad_uy[b, p, 1]
                    
                    # PDE residual: ∇²u + λu - f = 0
                    x_coord = X[b, p, 0]
                    y_coord = X[b, p, 1]
                    f_val = self.f(x_coord, y_coord)
                    
                    residual = u_xx + u_yy + self.lam * u_point - f_val
                    errs.append(residual ** 2)
        
        if len(errs) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Return mean of squared residuals
        return torch.stack(errs).mean()
    
    def pde_residual_loss_vectorized(self, model, batch):
        """
        Alternative vectorized version - faster but may have gradient issues.
        """
        device = self.device
        P = self.patch_size
        
        # Gather patches
        coords_list = []
        for coords in batch.get("X_f", []):
            coords_list.append(coords)
        
        if len(coords_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Process all at once
        all_coords = torch.stack(coords_list, dim=0).to(device)  # [N,P,2]
        N = all_coords.shape[0]
        
        # Flatten for easier gradient computation
        X_flat = all_coords.reshape(-1, 2)  # [N*P, 2]
        X_flat = X_flat.clone().detach().requires_grad_(True)
        
        # Reshape back for model
        X = X_flat.reshape(N, P, 2)
        
        # Forward
        U = model(X)  # [N,P,1] or [N,P]
        if U.dim() == 2:
            U = U.unsqueeze(-1)
        U_flat = U.reshape(-1, 1)  # [N*P, 1]
        
        # Compute gradients with sum trick
        grad_outputs = torch.ones_like(U_flat)
        grad_u = torch.autograd.grad(
            outputs=U_flat,
            inputs=X_flat,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]  # [N*P, 2]
        
        u_x = grad_u[:, 0:1]  # [N*P, 1]
        u_y = grad_u[:, 1:2]  # [N*P, 1]
        
        # Second derivatives
        grad_ux = torch.autograd.grad(
            outputs=u_x,
            inputs=X_flat,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        grad_uy = torch.autograd.grad(
            outputs=u_y,
            inputs=X_flat,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_xx = grad_ux[:, 0:1]  # [N*P, 1]
        u_yy = grad_uy[:, 1:2]  # [N*P, 1]
        
        # PDE residual
        f_vals = self.f(X_flat[:, 0], X_flat[:, 1]).unsqueeze(-1)  # [N*P, 1]
        residual = u_xx + u_yy + self.lam * U_flat - f_vals
        
        return (residual ** 2).mean()

    def boundary_loss(self, model, batch):
        """
        FIXED: Compute boundary loss ensuring gradients flow.
        """
        device = self.device
        
        coords_list = []
        bmask_list = []
        ub_list = []
        
        for i, patch in enumerate(batch.get("X_b", [])):
            coords = patch["coords"]
            bmask = patch["boundary_mask"]
            coords_list.append(coords)
            bmask_list.append(bmask)
            
            if batch.get("u_b") and i < len(batch["u_b"]):
                ub_list.append(batch["u_b"][i])
            else:
                ub_list.append(None)
        
        if len(coords_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        errs = []
        
        for coords, bmask, ub in zip(coords_list, bmask_list, ub_list):
            coords = coords.to(device)
            bmask = bmask.to(device)
            
            # Forward through model - keep gradients
            pred = model(coords)  # [P,1] or [P]
            if pred.dim() > 1:
                pred = pred.squeeze(-1)  # [P]
            
            # Get boundary points
            pred_b = pred[bmask]  # [Nb]
            
            if ub is not None:
                ub = ub.to(device)
                if ub.dim() > 1:
                    ub = ub.squeeze(-1)
                # Get boundary values, filtering NaNs
                ub_b = ub[bmask]
                valid = ~torch.isnan(ub_b)
                if valid.any():
                    errs.append((pred_b[valid] - ub_b[valid]) ** 2)
            else:
                # Use exact solution
                coords_b = coords[bmask]
                ub_b = self.u_star(coords_b[:, 0], coords_b[:, 1])
                errs.append((pred_b - ub_b) ** 2)
        
        if len(errs) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Concatenate and return mean
        all_errs = torch.cat([e.reshape(-1) for e in errs])
        return all_errs.mean()

    def initial_loss(self, model, batch):
        """No initial condition for steady-state problems."""
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def relative_l2_on_grid(self, model, grid_cfg):
        """Compute relative L2 error on grid."""
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        device = self.device
        
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, device)
        
        with torch.no_grad():
            # Process in patches
            px, py = self.px, self.py
            U_pred = torch.zeros(nx, ny, device=device)
            
            for i in range(0, nx - px + 1, px):
                for j in range(0, ny - py + 1, py):
                    # Extract patch
                    x_patch = Xg[i:i+px, j:j+py].reshape(-1)
                    y_patch = Yg[i:i+px, j:j+py].reshape(-1)
                    coords = torch.stack([x_patch, y_patch], dim=1)
                    
                    # Predict
                    u_patch = model(coords).squeeze(-1).reshape(px, py)
                    U_pred[i:i+px, j:j+py] = u_patch
            
            # True solution
            U_true = self.u_star(Xg, Yg)
            
            # Relative L2 error
            num = torch.linalg.norm((U_pred - U_true).reshape(-1))
            den = torch.linalg.norm(U_true.reshape(-1))
            return (num / den).item()

    def plot_final(self, model, grid_cfg, out_dir):
        """Generate final plots."""
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, self.device)
        
        with torch.no_grad():
            # Similar to relative_l2_on_grid
            U_pred = torch.zeros(nx, ny, device=self.device)
            px, py = self.px, self.py
            
            for i in range(0, nx - px + 1, px):
                for j in range(0, ny - py + 1, py):
                    x_patch = Xg[i:i+px, j:j+py].reshape(-1)
                    y_patch = Yg[i:i+px, j:j+py].reshape(-1)
                    coords = torch.stack([x_patch, y_patch], dim=1)
                    u_patch = model(coords).squeeze(-1).reshape(px, py)
                    U_pred[i:i+px, j:j+py] = u_patch
            
            U_true = self.u_star(Xg, Yg)
        
        return save_plots_2d(
            Xg.detach().cpu().numpy(),
            Yg.detach().cpu().numpy(),
            U_true.detach().cpu().numpy(),
            U_pred.detach().cpu().numpy(),
            out_dir,
            "helmholtz2d_steady_patch"
        )