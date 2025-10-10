"""
helmholtz2d_patch.py - Time-dependent 2D Helmholtz equation with patch-based sampling
Wave equation: ∂²u/∂t² - c²∇²u + λu = f(x,y,t)
Fixed version with proper gradient tracking and batch handling.
"""

import math
import numpy as np
import torch
from pinnlab.experiments.base_patch import BaseExperiment_Patch
from pinnlab.data.geometries import linspace_2d
from pinnlab.data.patches import extract_xy_patches, attach_time
from pinnlab.utils.plotting import save_plots_2d


class Helmholtz2D_patch(BaseExperiment_Patch):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        
        # Domain bounds
        self.xa, self.xb = cfg["domain"]["x"]
        self.ya, self.yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]
        
        # Patch configuration
        self.px = cfg["patch"]["x"]  # points per patch in x
        self.py = cfg["patch"]["y"]  # points per patch in y
        self.pt = cfg["patch"].get("t", 3)  # points per patch in time
        self.patch_size = self.px * self.py * self.pt
        
        # Grid configuration
        self.gx = cfg["grid"]["x"]
        self.gy = cfg["grid"]["y"]
        self.gt = cfg["grid"]["t"]
        
        # Compute grid spacings for finite differences
        self.dx = (self.xb - self.xa) / (self.gx - 1)
        self.dy = (self.yb - self.ya) / (self.gy - 1)
        self.dt = (self.t1 - self.t0) / (self.gt - 1)
        
        # Stride
        self.sx = cfg.get("stride", {}).get("x", self.px)
        self.sy = cfg.get("stride", {}).get("y", self.py)
        self.st = cfg.get("stride", {}).get("t", self.pt)
        
        # PDE parameters
        self.c = float(cfg.get("wave_speed", 1.0))
        self.lam = float(cfg.get("lambda", 0.0))
        self.a1 = float(cfg.get("a1", 1.0))
        self.a2 = float(cfg.get("a2", 1.0))
        self.omega = float(cfg.get("omega", 2.0))
        
        # Padding modes
        self.pad_mode_xy = cfg.get("pad_mode", {}).get("xy", "zero")
        self.pad_mode_t = cfg.get("pad_mode", {}).get("t", "zero")
        
        self.batch_size = cfg.get("batch_size", 16)
        self.input_dim = 3
        
        # Finite difference scheme
        self.fd_scheme = cfg.get("fd_scheme", "central")  # central, forward, backward
        
        print(f"Using finite differences with dx={self.dx:.4f}, dy={self.dy:.4f}, dt={self.dt:.4f}")
        
    def u_star(self, x, y, t):
        """Exact solution."""
        return torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y) * torch.cos(self.omega * math.pi * t)
    
    def f(self, x, y, t):
        """Source term."""
        k2 = (self.a1**2 + self.a2**2) * (math.pi**2)
        omega2 = (self.omega * math.pi)**2
        coeff = -omega2 + self.c**2 * k2 + self.lam
        return coeff * self.u_star(x, y, t)
    
    def u0(self, x, y):
        """Initial condition."""
        return self.u_star(x, y, torch.zeros_like(x))
    
    def ut0(self, x, y):
        """Initial velocity."""
        return torch.zeros_like(x)
    
    def g_boundary(self, x, y, t):
        """Boundary condition."""
        return self.u_star(x, y, t)
    
    def sample_patches(self):
        """Sample patches in space-time domain."""
        spatial_patches = extract_xy_patches(
            self.xa, self.xb, self.ya, self.yb,
            self.gx, self.gy, self.px, self.py,
            self.sx, self.sy, self.pad_mode_xy, self.device
        )
        
        patches_3d = attach_time(
            spatial_patches,
            self.t0, self.t1, self.gt,
            self.pt, self.st, self.pad_mode_t,
            sample_mode="sliding"
        )
        
        return patches_3d
    
    def sample_batch(self, *_, **__):
        """Sample batch compatible with training loop."""
        patches = self.sample_patches()
        return {
            "patches": patches,
            "X_f": patches["coords"],
            "X_b": None,
            "X_0": None
        }
    
    def _ensure_boolean(self, tensor):
        """Convert to boolean if needed."""
        return tensor > 0.5 if tensor.dtype != torch.bool else tensor
    
    def _compute_finite_differences_on_patch(self, u_patch, dx, dy, dt):
        """
        Compute finite difference derivatives on a structured patch.
        
        Args:
            u_patch: [px, py, pt] tensor of u values on the patch grid
            dx, dy, dt: grid spacings
            
        Returns:
            u_xx, u_yy, u_tt at interior points
        """
        px, py, pt = u_patch.shape
        
        # For second derivatives, we need at least 3 points in each dimension
        if px < 3 or py < 3 or pt < 3:
            # Return zeros for edge cases
            return (torch.zeros(1, device=u_patch.device),
                   torch.zeros(1, device=u_patch.device),
                   torch.zeros(1, device=u_patch.device),
                   torch.zeros(1, 3, device=u_patch.device))
        
        # Central differences for second derivatives
        # ∂²u/∂x² ≈ (u[i+1,j,k] - 2*u[i,j,k] + u[i-1,j,k]) / dx²
        u_xx = (u_patch[2:, 1:-1, 1:-1] - 2*u_patch[1:-1, 1:-1, 1:-1] + u_patch[:-2, 1:-1, 1:-1]) / (dx**2)
        
        # ∂²u/∂y² ≈ (u[i,j+1,k] - 2*u[i,j,k] + u[i,j-1,k]) / dy²
        u_yy = (u_patch[1:-1, 2:, 1:-1] - 2*u_patch[1:-1, 1:-1, 1:-1] + u_patch[1:-1, :-2, 1:-1]) / (dy**2)
        
        # ∂²u/∂t² ≈ (u[i,j,k+1] - 2*u[i,j,k] + u[i,j,k-1]) / dt²
        u_tt = (u_patch[1:-1, 1:-1, 2:] - 2*u_patch[1:-1, 1:-1, 1:-1] + u_patch[1:-1, 1:-1, :-2]) / (dt**2)
        
        # Get interior coordinates (where all derivatives are defined)
        interior_coords = []
        for i in range(1, px-1):
            for j in range(1, py-1):
                for k in range(1, pt-1):
                    interior_coords.append([i, j, k])
        
        interior_coords = torch.tensor(interior_coords, device=u_patch.device)
        
        # Flatten the derivatives
        u_xx = u_xx.reshape(-1)
        u_yy = u_yy.reshape(-1)
        u_tt = u_tt.reshape(-1)
        
        return u_xx, u_yy, u_tt, interior_coords
    
    def pde_residual_loss(self, model, batch):
        """PDE residual loss using finite differences."""
        device = self.device
        
        if "patches" not in batch:
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        patches = batch["patches"]
        coords = patches["coords"]  # [L*Nt, P*kt, 3]
        valid = self._ensure_boolean(patches["valid"])
        is_bnd = self._ensure_boolean(patches["is_bnd"])
        is_ic = self._ensure_boolean(patches["is_ic"])
        
        # Process patches
        total_patches = coords.size(0)
        all_losses = []
        
        for patch_idx in range(min(total_patches, self.batch_size)):
            patch_coords = coords[patch_idx]  # [P*kt, 3]
            patch_valid = valid[patch_idx]
            patch_bnd = is_bnd[patch_idx]
            patch_ic = is_ic[patch_idx]
            
            # Skip if patch doesn't have enough interior points
            if not patch_valid.any():
                continue
            
            # Reshape patch to grid structure [px, py, pt]
            # Assuming the patch is ordered correctly
            patch_coords_grid = patch_coords.reshape(self.px, self.py, self.pt, 3)
            
            # Get the patch bounds to create a local grid
            x_min = patch_coords_grid[:, :, :, 0].min()
            x_max = patch_coords_grid[:, :, :, 0].max()
            y_min = patch_coords_grid[:, :, :, 1].min()
            y_max = patch_coords_grid[:, :, :, 1].max()
            t_min = patch_coords_grid[:, :, :, 2].min()
            t_max = patch_coords_grid[:, :, :, 2].max()
            
            # Create stencil for finite differences
            # We need to evaluate at grid points for the stencil
            x_grid = torch.linspace(x_min, x_max, self.px, device=device)
            y_grid = torch.linspace(y_min, y_max, self.py, device=device)
            t_grid = torch.linspace(t_min, t_max, self.pt, device=device)
            
            # Create meshgrid for evaluation
            X_grid, Y_grid, T_grid = torch.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
            
            # Flatten and evaluate model at all grid points
            grid_points = torch.stack([
                X_grid.reshape(-1),
                Y_grid.reshape(-1),
                T_grid.reshape(-1)
            ], dim=1)  # [px*py*pt, 3]
            
            # Ensure grid_points requires grad for backprop
            grid_points = grid_points.requires_grad_(True)
            
            # Forward through model
            u_flat = model(grid_points.unsqueeze(0)).squeeze(0)  # [px*py*pt, 1]
            if u_flat.dim() > 1:
                u_flat = u_flat.squeeze(-1)
            
            # Reshape to grid
            u_patch = u_flat.reshape(self.px, self.py, self.pt)
            
            # Compute finite differences
            u_xx, u_yy, u_tt, interior_indices = self._compute_finite_differences_on_patch(
                u_patch, self.dx, self.dy, self.dt
            )
            
            if len(interior_indices) == 0:
                continue
            
            # Get interior coordinates for source term evaluation
            interior_coords_list = []
            for idx in interior_indices:
                i, j, k = idx[0].item(), idx[1].item(), idx[2].item()
                interior_coords_list.append([
                    X_grid[i, j, k],
                    Y_grid[i, j, k],
                    T_grid[i, j, k]
                ])
            
            if len(interior_coords_list) > 0:
                interior_coords_tensor = torch.stack([torch.stack(c) for c in interior_coords_list])
                
                # Get u values at interior points
                u_interior = []
                for idx in interior_indices:
                    i, j, k = idx[0].item(), idx[1].item(), idx[2].item()
                    u_interior.append(u_patch[i, j, k])
                u_interior = torch.stack(u_interior)
                
                # Evaluate source term
                f_vals = self.f(
                    interior_coords_tensor[:, 0],
                    interior_coords_tensor[:, 1],
                    interior_coords_tensor[:, 2]
                )
                
                # PDE residual: ∂²u/∂t² - c²∇²u + λu - f = 0
                laplacian = u_xx + u_yy
                residual = u_tt - self.c**2 * laplacian + self.lam * u_interior - f_vals
                
                all_losses.append((residual ** 2).mean())
        
        if len(all_losses) == 0:
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        return torch.stack(all_losses).mean()
    
    def boundary_loss(self, model, batch):
        """Boundary loss - still uses direct evaluation."""
        device = self.device
        
        if "patches" not in batch:
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        patches = batch["patches"]
        coords = patches["coords"]
        valid = self._ensure_boolean(patches["valid"])
        is_bnd = self._ensure_boolean(patches["is_bnd"])
        
        bnd_mask = valid & is_bnd
        
        if not bnd_mask.any():
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        bnd_coords = coords[bnd_mask]
        n_bnd = len(bnd_coords)
        all_losses = []
        
        for i in range(0, n_bnd, self.patch_size):
            batch_end = min(i + self.patch_size, n_bnd)
            batch_coords = bnd_coords[i:batch_end]
            
            # Pad if needed
            if len(batch_coords) < self.patch_size:
                pad_size = self.patch_size - len(batch_coords)
                pad_coords = torch.zeros(pad_size, 3, device=device)
                batch_coords_padded = torch.cat([batch_coords, pad_coords], dim=0)
                actual_size = batch_end - i
            else:
                batch_coords_padded = batch_coords
                actual_size = self.patch_size
            
            # Ensure gradient tracking
            batch_coords_padded = batch_coords_padded.requires_grad_(True)
            
            # Forward
            u_pred = model(batch_coords_padded.unsqueeze(0)).squeeze(0)
            if u_pred.dim() > 1:
                u_pred = u_pred[:actual_size, 0]
            else:
                u_pred = u_pred[:actual_size]
            
            # True values
            x_b = batch_coords[:actual_size, 0]
            y_b = batch_coords[:actual_size, 1]
            t_b = batch_coords[:actual_size, 2]
            u_true = self.g_boundary(x_b, y_b, t_b)
            
            all_losses.append(((u_pred - u_true) ** 2).mean())
        
        if len(all_losses) == 0:
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        return torch.stack(all_losses).mean()
    
    def initial_loss(self, model, batch):
        """Initial condition loss using finite differences for velocity."""
        device = self.device
        
        if "patches" not in batch:
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        patches = batch["patches"]
        coords = patches["coords"]
        valid = self._ensure_boolean(patches["valid"])
        is_ic = self._ensure_boolean(patches["is_ic"])
        
        ic_mask = valid & is_ic
        
        if not ic_mask.any():
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        # For initial conditions, we need to evaluate at t=0 and t=dt
        ic_coords = coords[ic_mask]
        n_ic = len(ic_coords)
        all_losses_u0 = []
        all_losses_ut0 = []
        
        for i in range(0, n_ic, self.patch_size):
            batch_end = min(i + self.patch_size, n_ic)
            batch_coords = ic_coords[i:batch_end]
            
            # Evaluate at t=0
            coords_t0 = batch_coords.clone()
            coords_t0[:, 2] = self.t0
            
            # Evaluate at t=dt for finite difference of ∂u/∂t
            coords_t1 = batch_coords.clone()
            coords_t1[:, 2] = self.t0 + self.dt
            
            # Pad if needed
            if len(coords_t0) < self.patch_size:
                pad_size = self.patch_size - len(coords_t0)
                pad_coords = torch.zeros(pad_size, 3, device=device)
                coords_t0_padded = torch.cat([coords_t0, pad_coords], dim=0)
                coords_t1_padded = torch.cat([coords_t1, pad_coords], dim=0)
                actual_size = batch_end - i
            else:
                coords_t0_padded = coords_t0
                coords_t1_padded = coords_t1
                actual_size = self.patch_size
            
            # Ensure gradient tracking
            coords_t0_padded = coords_t0_padded.requires_grad_(True)
            coords_t1_padded = coords_t1_padded.requires_grad_(True)
            
            # Forward at t=0
            u_t0 = model(coords_t0_padded.unsqueeze(0)).squeeze(0)
            if u_t0.dim() > 1:
                u_t0 = u_t0[:actual_size, 0]
            else:
                u_t0 = u_t0[:actual_size]
            
            # Forward at t=dt
            u_t1 = model(coords_t1_padded.unsqueeze(0)).squeeze(0)
            if u_t1.dim() > 1:
                u_t1 = u_t1[:actual_size, 0]
            else:
                u_t1 = u_t1[:actual_size]
            
            # Initial value loss
            x_ic = batch_coords[:actual_size, 0]
            y_ic = batch_coords[:actual_size, 1]
            u0_true = self.u0(x_ic, y_ic)
            all_losses_u0.append(((u_t0 - u0_true) ** 2).mean())
            
            # Initial velocity loss using forward difference
            # ∂u/∂t ≈ (u(t+dt) - u(t)) / dt
            u_t_approx = (u_t1 - u_t0) / self.dt
            ut0_true = self.ut0(x_ic, y_ic)
            all_losses_ut0.append(((u_t_approx - ut0_true) ** 2).mean())
        
        if len(all_losses_u0) == 0:
            return torch.tensor(1e-10, device=device, requires_grad=True)
        
        loss_u0 = torch.stack(all_losses_u0).mean()
        loss_ut0 = torch.stack(all_losses_ut0).mean() if all_losses_ut0 else torch.tensor(0.0, device=device)
        
        return loss_u0 + loss_ut0
    
    def relative_l2_on_grid(self, model, grid_cfg):
        """Compute relative L2 error."""
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        device = self.device
        t_eval = self.t1
        
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, device)
        Tg = torch.full_like(Xg, t_eval)
        
        with torch.no_grad():
            U_pred = torch.zeros(nx, ny, device=device)
            
            for i in range(0, nx, self.px):
                for j in range(0, ny, self.py):
                    i_end = min(i + self.px, nx)
                    j_end = min(j + self.py, ny)
                    
                    x_patch = Xg[i:i_end, j:j_end].reshape(-1)
                    y_patch = Yg[i:i_end, j:j_end].reshape(-1)
                    spatial_size = len(x_patch)
                    
                    # Create 3D patch
                    coords = []
                    for _ in range(self.pt):
                        t_patch = torch.full((spatial_size,), t_eval, device=device)
                        coords.append(torch.stack([x_patch, y_patch, t_patch], dim=1))
                    coords = torch.cat(coords, dim=0)
                    
                    # Pad if needed
                    if len(coords) < self.patch_size:
                        pad_size = self.patch_size - len(coords)
                        coords = torch.cat([coords, torch.zeros(pad_size, 3, device=device)], dim=0)
                    
                    u_patch = model(coords.unsqueeze(0)).squeeze()
                    if u_patch.dim() > 1:
                        u_patch = u_patch[:, 0]
                    
                    # Average over time dimension
                    u_spatial = u_patch[:spatial_size * self.pt].reshape(self.pt, spatial_size).mean(dim=0)
                    U_pred[i:i_end, j:j_end] = u_spatial.reshape(i_end - i, j_end - j)
            
            U_true = self.u_star(Xg, Yg, Tg)
            num = torch.linalg.norm((U_pred - U_true).reshape(-1))
            den = torch.linalg.norm(U_true.reshape(-1))
            return (num / (den + 1e-8)).item()
    
    def plot_final(self, model, grid_cfg, out_dir):
        """Generate plots."""
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        nt = 4
        
        time_points = torch.linspace(self.t0, self.t1, nt, device=self.device)
        Xg, Yg = linspace_2d(self.xa, self.xb, self.ya, self.yb, nx, ny, self.device)
        
        all_figs = {}
        
        with torch.no_grad():
            for t_idx, t_eval in enumerate(time_points):
                Tg = torch.full_like(Xg, t_eval)
                U_pred = torch.zeros(nx, ny, device=self.device)
                
                for i in range(0, nx, self.px):
                    for j in range(0, ny, self.py):
                        i_end = min(i + self.px, nx)
                        j_end = min(j + self.py, ny)
                        
                        x_patch = Xg[i:i_end, j:j_end].reshape(-1)
                        y_patch = Yg[i:i_end, j:j_end].reshape(-1)
                        spatial_size = len(x_patch)
                        
                        coords = []
                        for _ in range(self.pt):
                            t_patch = torch.full((spatial_size,), t_eval, device=self.device)
                            coords.append(torch.stack([x_patch, y_patch, t_patch], dim=1))
                        coords = torch.cat(coords, dim=0)
                        
                        if len(coords) < self.patch_size:
                            pad_size = self.patch_size - len(coords)
                            coords = torch.cat([coords, torch.zeros(pad_size, 3, device=self.device)], dim=0)
                        
                        u_patch = model(coords.unsqueeze(0)).squeeze()
                        if u_patch.dim() > 1:
                            u_patch = u_patch[:, 0]
                        
                        u_spatial = u_patch[:spatial_size * self.pt].reshape(self.pt, spatial_size).mean(dim=0)
                        U_pred[i:i_end, j:j_end] = u_spatial.reshape(i_end - i, j_end - j)
                
                U_true = self.u_star(Xg, Yg, Tg)
                
                figs = save_plots_2d(
                    Xg.detach().cpu().numpy(),
                    Yg.detach().cpu().numpy(),
                    U_true.detach().cpu().numpy(),
                    U_pred.detach().cpu().numpy(),
                    out_dir,
                    f"helmholtz2d_t{t_idx:02d}"
                )
                all_figs.update(figs)
        
        return all_figs