import math
import numpy as np
import torch
import torch.nn.functional as F
from pinnlab.experiments.base import BaseExperiment
from pinnlab.data.geometries import linspace_2d
from pinnlab.data.samplers import sample_patches_2d_steady
from pinnlab.utils.plotting import save_plots_2d


class Helmholtz2DSteady(BaseExperiment):
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
        
        # Optimization settings
        self.derivative_method = cfg.get("derivative_method", "autograd")  # "autograd" or "finite_diff"
        
        # Precompute grid spacings for finite differences
        if self.derivative_method == "finite_diff":
            self.dx = (self.xb - self.xa) / (self.gx - 1)
            self.dy = (self.yb - self.ya) / (self.gy - 1)
            print(f"Using finite differences with dx={self.dx:.4f}, dy={self.dy:.4f}")
            
            # Precompute finite difference stencil masks for interior points
            self._compute_fd_masks()
        else:
            print("Using automatic differentiation")

    def _compute_fd_masks(self):
        """Precompute which points in a patch can use finite differences."""
        # For a px x py patch, interior points are those not on patch edges
        # Create a mask for valid FD points (need neighbors in all directions)
        self.fd_valid_mask = torch.zeros(self.px, self.py, dtype=torch.bool, device=self.device)
        if self.px > 2 and self.py > 2:
            self.fd_valid_mask[1:-1, 1:-1] = True
        self.fd_valid_mask_flat = self.fd_valid_mask.reshape(-1)
        
        # Precompute index offsets for finite difference stencil
        # For 2D Laplacian, we need: center, left, right, up, down
        idx = torch.arange(self.patch_size, device=self.device).reshape(self.px, self.py)
        
        # Store indices for finite difference computation
        self.fd_indices = {
            'center': idx[1:-1, 1:-1].reshape(-1) if self.px > 2 and self.py > 2 else torch.tensor([], device=self.device),
            'left':   idx[:-2, 1:-1].reshape(-1) if self.px > 2 and self.py > 2 else torch.tensor([], device=self.device),
            'right':  idx[2:, 1:-1].reshape(-1) if self.px > 2 and self.py > 2 else torch.tensor([], device=self.device),
            'up':     idx[1:-1, :-2].reshape(-1) if self.px > 2 and self.py > 2 else torch.tensor([], device=self.device),
            'down':   idx[1:-1, 2:].reshape(-1) if self.px > 2 and self.py > 2 else torch.tensor([], device=self.device),
        }

    def u_star(self, x, y):
        """Exact solution for the Helmholtz equation."""
        return torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y)
    
    def f(self, x, y):
        """Source term for the Helmholtz equation."""
        coeff = (-(self.a1**2 + self.a2**2) * (math.pi**2) + self.lam)
        return coeff * self.u_star(x, y)

    def sample_patches(self):
        """Sample patches with boundary conditions."""
        def g_dirichlet(x, y):
            return self.u_star(x.squeeze(-1), y.squeeze(-1))

        out = sample_patches_2d_steady(
            self.xa, self.xb,
            self.ya, self.yb,
            self.px, self.py,
            self.gx, self.gy,
            device=self.device,
            stride_x=None,
            stride_y=None,
            boundary_fn=g_dirichlet,
        )
        x_f, x_b, u_b = out["interior_patches"], out["boundary_patches"], out["true_boundary"]
        return x_f, x_b, u_b
    
    def sample_batch(self, *_, **__):
        x_f, x_b, u_b = self.sample_patches()
        return {"X_f": x_f, "X_b": x_b, "u_b": u_b}
    
    def pde_residual_loss_finite_diff(self, model, batch):
        """PDE residual using finite differences for Laplacian."""
        device = self.device
        coords_list = []
        
        # Collect interior patches
        for coords in batch.get("X_f", []):
            coords_list.append(coords)
        
        # Add interior points from boundary patches
        for patch in batch.get("X_b", []):
            coords = patch["coords"]
            bmask = patch["boundary_mask"]
            # For FD, we need the whole patch structure
            # Mark this patch as having boundary points
            coords_list.append(coords)
        
        if len(coords_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        all_losses = []
        bs = self.batch_size
        
        for start in range(0, len(coords_list), bs):
            chunk = coords_list[start:start+bs]
            if not chunk:
                continue
                
            # Stack patches
            C = torch.stack(chunk, dim=0).to(device)  # [B, P, 2]
            B = C.size(0)
            
            # Forward through model
            U = model(C)  # [B, P, 1] or [B, P]
            if U.dim() == 2:
                U = U.unsqueeze(-1)  # [B, P, 1]
            
            # Reshape to grid for finite differences
            U_grid = U.reshape(B, self.px, self.py)  # [B, px, py]
            
            # Compute Laplacian using finite differences (5-point stencil)
            # Only for interior points of each patch
            if len(self.fd_indices['center']) > 0:
                u_center = U_grid[:, 1:-1, 1:-1]  # [B, px-2, py-2]
                u_left   = U_grid[:, :-2, 1:-1]
                u_right  = U_grid[:, 2:, 1:-1]
                u_up     = U_grid[:, 1:-1, :-2]
                u_down   = U_grid[:, 1:-1, 2:]
                
                # Laplacian
                laplacian = (u_left - 2*u_center + u_right) / (self.dx**2) + \
                           (u_up - 2*u_center + u_down) / (self.dy**2)
                
                # Get coordinates for interior points
                X_interior = C[:, self.fd_indices['center'], :]  # [B, n_interior, 2]
                
                # Source term at interior points
                f_vals = self.f(X_interior[..., 0], X_interior[..., 1])  # [B, n_interior]
                
                # PDE residual: ∇²u + λu - f = 0
                u_center_flat = u_center.reshape(B, -1)  # [B, n_interior]
                residual = laplacian.reshape(B, -1) + self.lam * u_center_flat - f_vals
                
                # Loss
                loss = (residual ** 2).mean()
                all_losses.append(loss)
        
        if all_losses:
            return torch.stack(all_losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def pde_residual_loss_autograd(self, model, batch):
        """Original autograd-based PDE residual (vectorized version)."""
        device = self.device
        P = self.patch_size
        
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
            keep_mask_list.append(~bmask)
        
        if len(coords_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Process all at once (vectorized)
        all_coords = torch.stack(coords_list, dim=0).to(device)  # [N,P,2]
        all_masks = torch.stack(keep_mask_list, dim=0).to(device)  # [N,P]
        
        N = all_coords.shape[0]
        
        # Flatten for gradient computation
        X_flat = all_coords.reshape(-1, 2)  # [N*P, 2]
        mask_flat = all_masks.reshape(-1)  # [N*P]
        X_flat = X_flat.clone().detach().requires_grad_(True)
        
        # Reshape back for model
        X = X_flat.reshape(N, P, 2)
        
        # Forward
        U = model(X)
        if U.dim() == 2:
            U = U.unsqueeze(-1)
        U_flat = U.reshape(-1, 1)  # [N*P, 1]
        
        # First derivatives
        grad_u = torch.autograd.grad(
            outputs=U_flat,
            inputs=X_flat,
            grad_outputs=torch.ones_like(U_flat),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
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
        
        u_xx = grad_ux[:, 0:1]
        u_yy = grad_uy[:, 1:2]
        
        # PDE residual
        f_vals = self.f(X_flat[:, 0], X_flat[:, 1]).unsqueeze(-1)
        residual = u_xx + u_yy + self.lam * U_flat - f_vals
        
        # Apply mask and compute loss
        residual_masked = residual[mask_flat]
        return (residual_masked ** 2).mean()
    
    def pde_residual_loss(self, model, batch):
        """Dispatch to appropriate PDE residual computation."""
        if self.derivative_method == "finite_diff":
            return self.pde_residual_loss_finite_diff(model, batch)
        else:
            return self.pde_residual_loss_autograd(model, batch)

    def boundary_loss(self, model, batch):
        """Boundary loss computation (unchanged)."""
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
        
        # Vectorized processing
        all_coords = torch.stack(coords_list, dim=0).to(device)  # [N, P, 2]
        all_bmask = torch.stack(bmask_list, dim=0).to(device)  # [N, P]
        
        # Forward
        pred = model(all_coords)  # [N, P, 1] or [N, P]
        if pred.dim() > 2:
            pred = pred.squeeze(-1)  # [N, P]
        
        # Extract boundary values
        pred_b = pred[all_bmask]  # [total_boundary_points]
        
        # Compute target values
        coords_b = all_coords[all_bmask]  # [total_boundary_points, 2]
        ub_exact = self.u_star(coords_b[:, 0], coords_b[:, 1])
        
        loss = (pred_b - ub_exact) ** 2
        return loss.mean()

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
                    x_patch = Xg[i:i+px, j:j+py].reshape(-1)
                    y_patch = Yg[i:i+px, j:j+py].reshape(-1)
                    coords = torch.stack([x_patch, y_patch], dim=1)
                    
                    u_patch = model(coords).squeeze(-1).reshape(px, py)
                    U_pred[i:i+px, j:j+py] = u_patch
            
            U_true = self.u_star(Xg, Yg)
            
            num = torch.linalg.norm((U_pred - U_true).reshape(-1))
            den = torch.linalg.norm(U_true.reshape(-1))
            return (num / den).item()

    def plot_final(self, model, grid_cfg, out_dir):
        """Generate final plots."""
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, self.device)
        
        with torch.no_grad():
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