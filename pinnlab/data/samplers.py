import torch
from typing import Callable, Dict, List, Optional, Tuple, Union

def concat_time(X, t):
    if t is None: return X
    if X.dim()==1: X = X[:,None]
    if t.dim()==1: t = t[:,None]
    if X.shape[0] != t.shape[0]:
        t = t.expand(X.shape[0], -1)
    return torch.cat([X, t], dim=1)

def uniform_time(n, t0, t1, device):
    return torch.rand(n,1,device=device)*(t1-t0)+t0

def sample_patches_2d_steady(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    patch_x: int, patch_y: int,
    grid_x: int, grid_y: int,
    *,
    device: Union[str, torch.device] = "cpu",
    stride_x: Optional[int] = None,
    stride_y: Optional[int] = None,
    boundary_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]]]:
    """
    Patch sampler on a rectangular 2D domain.

    Parameters
    ----------
    patch_x, patch_y : int
        Number of points per patch in each dimension. 
        A patch with patch_x=3, patch_y=3 contains 3×3=9 points.
    grid_x, grid_y : int
        Total number of grid points along each axis.
        E.g., grid_x=30 means 30 points along x-axis.

    Returns
    -------
    dict with keys:
        "interior_patches": List[Tensor], each [P, 2] where P = patch_x * patch_y
        "boundary_patches": List[Dict] with coords, boundary_mask, grid_idx
        "true_boundary": List[Tensor] or None, boundary values with NaN at interior
    """
    device = torch.device(device)

    # Create the global grid
    nx_pts = grid_x  # Number of points in x
    ny_pts = grid_y  # Number of points in y
    xs = torch.linspace(x_min, x_max, nx_pts, device=device)
    ys = torch.linspace(y_min, y_max, ny_pts, device=device)

    # Default stride (non-overlapping)
    if stride_x is None: stride_x = patch_x
    if stride_y is None: stride_y = patch_y

    interior_patches: List[torch.Tensor] = []
    boundary_patches: List[Dict[str, torch.Tensor]] = []
    boundary_values: Optional[List[torch.Tensor]] = [] if (boundary_fn is not None) else None

    # Slide patches over the grid points
    # Starting points for patches (in point indices)
    ix0s = list(range(0, nx_pts - patch_x + 1, stride_x))
    iy0s = list(range(0, ny_pts - patch_y + 1, stride_y))

    for ix0 in ix0s:
        # Extract patch_x consecutive points starting from ix0
        px_idx = torch.arange(ix0, ix0 + patch_x, device=device)  # [patch_x]
        
        for iy0 in iy0s:
            # Extract patch_y consecutive points starting from iy0
            py_idx = torch.arange(iy0, iy0 + patch_y, device=device)  # [patch_y]

            # Build the patch grid and flatten
            XX, YY = torch.meshgrid(xs[px_idx], ys[py_idx], indexing="ij")  # [patch_x, patch_y]
            coords = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)  # [P, 2] where P = patch_x * patch_y

            # Track global grid indices for boundary detection
            GX, GY = torch.meshgrid(px_idx, py_idx, indexing="ij")  # [patch_x, patch_y]
            gix = GX.reshape(-1).long()  # [P]
            giy = GY.reshape(-1).long()  # [P]
            grid_idx = torch.stack([gix, giy], dim=1)  # [P, 2]

            # FIXED: Boundary detection with correct max indices
            # Points on boundary have indices: 0 or (nx_pts-1) for x, 0 or (ny_pts-1) for y
            boundary_mask = (gix == 0) | (gix == nx_pts - 1) | (giy == 0) | (giy == ny_pts - 1)

            if torch.any(boundary_mask):
                # This patch contains at least one boundary point
                patch_rec: Dict[str, torch.Tensor] = {
                    "coords": coords,  # [P, 2]
                    "boundary_mask": boundary_mask,  # [P]
                    "grid_idx": grid_idx,  # [P, 2]
                }
                boundary_patches.append(patch_rec)

                if boundary_values is not None:
                    # Compute boundary values, NaN for interior points
                    ub = torch.full((coords.size(0), 1), float("nan"), device=device)
                    if boundary_mask.any():
                        xb = coords[boundary_mask, 0:1]
                        yb = coords[boundary_mask, 1:2]
                        vals = boundary_fn(xb, yb)  # Expected shape [Nb, 1]
                        if vals.dim() == 1: 
                            vals = vals[:, None]
                        ub[boundary_mask] = vals
                    boundary_values.append(ub)  # [P, 1] with NaNs at interior
            else:
                # Pure interior patch (no boundary points)
                interior_patches.append(coords)  # [P, 2]

    out: Dict[str, Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]]] = {
        "interior_patches": interior_patches,
        "boundary_patches": boundary_patches,
        "true_boundary": boundary_values,
    }
    return out