# pinnlab/data/samplers.py  (append this)

from typing import Callable, Dict, List, Optional, Tuple, Union
import torch

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
    x_min, x_max, y_min, y_max : float
        Domain bounds.
    patch_x, patch_y : int
        Patch size in *grid cells* (segments). A patch that is patch_x cells
        wide in x and patch_y cells tall in y contains (nx_pts) * (ny_pts)
        grid *points*, where nx_pts = patch_x + 1 (unless truncated at edges),
        and similarly for y.
    grid_x, grid_y : int
        Number of *equal divisions* per axis. E.g., grid_x=100 divides the x-axis
        into 100 segments → 101 grid points in x.
    device : str or torch.device
        Target device for returned tensors.
    stride_x, stride_y : Optional[int]
        Sliding stride in cells. Defaults to non-overlapping stride == patch size.
    boundary_fn : Optional[(x:Tensor, y:Tensor) -> Tensor]
        If provided, will be used to compute Dirichlet target values on *exact*
        boundary points. Must return shape [N,1] for N input points.

    Returns
    -------
    dict with keys:
        {
          "interior_patches": List[Tensor],  where each is [Pi, 2] (x,y)
          "boundary_patches": List[Dict],    where each dict contains:
                {
                  "coords": Tensor [Pb, 2],          # (x,y) of patch points
                  "boundary_mask": BoolTensor [Pb],  # True at boundary points
                  "grid_idx": LongTensor [Pb, 2],    # (ix, iy) w.r.t. global grid
                }
          "true_boundary":    List[Tensor] OR None,  # per-boundary-patch [Pb,1] with NaN at non-boundary points
        }

    Notes
    -----
    - grid_x, grid_y are *divisions* → number of grid points is (grid_x+1, grid_y+1).
    - A patch is categorized as a "boundary patch" if it contains *any* boundary point.
    - Boundary points inside each boundary patch are flagged via 'boundary_mask'.
    - If 'boundary_fn' is given, 'true_boundary' is a list of tensors aligned
      with 'boundary_patches' (NaN for non-boundary entries). Otherwise it's None.
    """
    device = torch.device(device)

    # Global grid (points = divisions + 1)
    nx_pts = grid_x + 1
    ny_pts = grid_y + 1
    xs = torch.linspace(x_min, x_max, nx_pts, device=device)
    ys = torch.linspace(y_min, y_max, ny_pts, device=device)

    # Default: stride = 1
    if stride_x is None: stride_x = 1
    if stride_y is None: stride_y = 1

    interior_patches: List[torch.Tensor] = []
    boundary_patches: List[Dict[str, torch.Tensor]] = []
    boundary_values: Optional[List[torch.Tensor]] = [] if (boundary_fn is not None) else None

    # Slide patches over the *cell* index space [0..grid_x-1] × [0..grid_y-1]
    # Each patch covers cells [ix0 .. ix1-1], points [ix0 .. ix1] (inclusive).
    ix0s = list(range(0, grid_x - patch_x + 2, stride_x))
    iy0s = list(range(0, grid_y - patch_y + 2, stride_y))

    for ix0 in ix0s:
        ix1 = ix0 + patch_x   # last cell index covered
        # corresponding *point* indices are inclusive
        px_idx = torch.arange(ix0, ix1, device=device)  # shape [nxp]
        for iy0 in iy0s:
            iy1 = iy0 + patch_y
            py_idx = torch.arange(iy0, iy1, device=device)  # shape [nyp]

            # Build the patch point grid → flatten
            XX, YY = torch.meshgrid(xs[px_idx], ys[py_idx], indexing="ij")   # [nxp, nyp]
            coords = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)    # [P,2]

            # print("---XX---")
            # print(XX)
            # print("---YY---")
            # print(YY)

            # Also keep global *grid* indices per point (for exact boundary tagging)
            GX, GY = torch.meshgrid(px_idx, py_idx, indexing="ij")           # [nxp, nyp]
            gix = GX.reshape(-1).long()                                      # [P]
            giy = GY.reshape(-1).long()
            grid_idx = torch.stack([gix, giy], dim=1)                         # [P,2]

            # Boundary mask: any point lying on the rectangle border in *global indices*
            #  (left/right edges are ix==0 or ix==grid_x; bottom/top are iy==0 or iy==grid_y)
            boundary_mask = (gix == 0) | (gix == grid_x) | (giy == 0) | (giy == grid_y)

            # print(boundary_mask)

            if torch.any(boundary_mask):
                patch_rec: Dict[str, torch.Tensor] = {
                    "coords": coords,                     # [P,2]
                    "boundary_mask": boundary_mask,       # [P]
                    "grid_idx": grid_idx,                 # [P,2]
                }
                boundary_patches.append(patch_rec)

                # print("---coords---")
                # print(coords)

                if boundary_values is not None:
                    # Fill NaN for non-boundary, compute true values on boundary
                    ub = torch.full((coords.size(0), 1), float("nan"), device=device)
                    if boundary_mask.any():
                        xb = coords[boundary_mask, 0:1]
                        # print("---xb---")
                        # print(xb)
                        yb = coords[boundary_mask, 1:2]
                        # print("---yb---")
                        # print(yb)
                        vals = boundary_fn(xb, yb)  # expected shape [Nb,1]
                        if vals.dim() == 1: vals = vals[:, None]
                        ub[boundary_mask] = vals
                    print("---ub---")
                    print(ub)
                    boundary_values.append(ub)  # [P,1] with NaNs at interior points
            else:
                # Pure interior patch
                interior_patches.append(coords)           # [P,2]

    print("---boundary_values---")
    print(boundary_values)

    out: Dict[str, Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]]] = {
        "interior_patches": interior_patches,
        "boundary_patches": boundary_patches,
        "true_boundary": boundary_values,  # List[[P,1]] with NaNs, or None if boundary_fn is None
    }
    return out

def g_dirichlet(x, y):
    # Example boundary condition u = sin(pi x) + cos(pi y)
    import math
    return torch.sin(math.pi * x) + torch.cos(math.pi * y)

batch = sample_patches_2d_steady(
    0.0, 1.0, 0.0, 1.0,
    patch_x=3, patch_y=3,
    grid_x=10, grid_y=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    stride_x=1,
    stride_y=1,
    boundary_fn=g_dirichlet
)

x_f = batch["interior_patches"]      # list of [P,2]
x_b = batch["boundary_patches"]      # list of {"coords":[P,2], "boundary_mask":[P], ...}
u_b = batch["true_boundary"]         # list of [P,1] (NaN for non-boundary)

