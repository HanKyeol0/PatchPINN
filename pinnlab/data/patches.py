import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def _build_xy_grid(xa, xb, ya, yb, nx, ny, device):
    # Xg, Yg: [nx, ny] (indexing='xy')
    x = torch.linspace(xa, xb, nx, device=device)
    y = torch.linspace(ya, yb, ny, device=device)
    Xg, Yg = torch.meshgrid(x, y, indexing="xy")
    # For unfold, arrange to [1, C=2, H=ny, W=nx]
    Ximg = Xg.T  # [ny, nx]
    Yimg = Yg.T  # [ny, nx]
    img  = torch.stack([Ximg, Yimg], dim=0).unsqueeze(0)
    mask = torch.ones((1, 1, ny, nx), device=device)
    return img, mask, (x, y)

def _pad2d(t, pad_hw, mode, value=0.0):
    _, _, ny, nx = t.shape
    # pad order for F.pad on 4D [N,C,H,W] is (W_left, W_right, H_top, H_bottom)
    (pw0, pw1), (ph0, ph1) = pad_hw
    if mode == "extend":
        return F.pad(t, (pw0, pw1, ph0, ph1), mode="constant", value=value)
    if mode in ("reflect", "replicate", "circular"):
        return F.pad(t, (pw0, pw1, ph0, ph1), mode=mode)
    if mode == "none":
        return t
    raise ValueError(f"Unknown pad mode: {mode}")

def extract_xy_patches(
    xa: float, xb: float, ya: float, yb: float,
    nx: int, ny: int,
    kx: int, ky: int,
    sx: int, sy: int,
    pad_mode: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      coords: [L, P, 2]  (x,y per point in each patch; L = #patches, P = kx*ky)
      valid:  [L, P]     (0 for padded points, 1 valid)
      is_bnd: [L, P]     (1 if that point is on domain boundary)
      meta:   dict with strides/sizes
    """
    # pad_xa = xa - (xb-xa)/(nx-1)
    # pad_xb = xb + (xb-xa)/(nx-1)
    # pad_ya = ya - (yb-ya)/(ny-1)
    # pad_yb = yb + (yb-ya)/(ny-1)
    # img, mask, (xv, yv) = _build_xy_grid(pad_xa, pad_xb, pad_ya, pad_yb, nx+2, ny+2, device) # with 1-cell pad
    img, mask, (xv, yv) = _build_xy_grid(xa, xb, ya, yb, nx, ny, device)

    unfold = torch.nn.Unfold(kernel_size=(ky, kx), stride=(sy, sx))

    # Unfold coordinates: [1, 2*P, L] -> [L, 2*P] # L means number of patches, and P = kx*ky
    patches_xy = unfold(img).squeeze(0).transpose(0, 1).contiguous() # [L, 2*P] (L patches, P=kx*ky)

    P = kx * ky
    L = patches_xy.shape[0]
    # Split channels: first P are X, next P are Y, then stack to [L, P, 2]
    xs = patches_xy[:, :P]
    ys = patches_xy[:, P:]
    coords = torch.stack([xs, ys], dim=-1)  # [L, P, 2] (left bottom -> right top)

    # Unfold mask: [1, 1*P, L] -> [L, P] (valid cells)
    patches_mask = unfold(mask).squeeze(0).transpose(0, 1).contiguous()
    valid = (patches_mask > 0.5).to(coords.dtype)

    # Mark boundary points (spatial only)
    eps_x = (xb - xa) / (nx - 1)
    eps_y = (yb - ya) / (ny - 1)
    x = coords[..., 0]; y = coords[..., 1]
    is_bnd = ((x - xa).abs() <= 0.5*eps_x) | ((x - xb).abs() <= 0.5*eps_x) | \
             ((y - ya).abs() <= 0.5*eps_y) | ((y - yb).abs() <= 0.5*eps_y)
    is_bnd = (is_bnd & (valid > 0)).to(coords.dtype)

    meta = {
        "L": L, "P": P, "kx": kx, "ky": ky, "sx": sx, "sy": sy,
        "nx": nx, "ny": ny,
    }
    return {"coords": coords, "valid": valid, "is_bnd": is_bnd, "meta": meta}

def attach_time(
    patches: Dict[str, torch.Tensor],
    t0: float, t1: float, nt: int,
    kt: int, st: int,
    pad_mode_t: str = "reflect",
):
    """
    Expand spatial patches over sliding time windows of length kt, stride st.
    Supports pad_mode_t in {"reflect","replicate","circular","constant","zero","none"}.

    Input (from extract_xy_patches):
      patches["coords"]: [L, P, 2]  (x,y)
      patches["valid"] : [L, P]
      patches["is_bnd"]: [L, P]
      patches["meta"]  : {...}

    Output:
      dict with keys: coords:[L*Nt, P*kt, 3], valid:[L*Nt, P*kt], is_bnd:[L*Nt, P*kt],
      is_ic:[L*Nt, P*kt], meta:{..., "kt","st","Nt","P3"}
    """
    coords = patches["coords"]     # [L,P,2]
    valid  = patches["valid"]      # [L,P]
    bnd    = patches["is_bnd"]     # [L,P]
    meta   = patches["meta"]
    device = coords.device
    dtype  = coords.dtype

    L, P, _ = coords.shape

    # time vector (length nt) -> shape (1,1,nt) for non-constant pad
    t = torch.linspace(t0, t1, nt, device=device, dtype=dtype)         # [nt]
    t3 = t.view(1, 1, -1)                                              # [1,1,nt]

    # "same"-style temporal padding (centered window)
    pL = kt // 2
    pR = (kt - 1) // 2

    # choose pad behavior
    pad_mode_t = (pad_mode_t or "none").lower()
    if pad_mode_t in {"reflect", "replicate", "circular"}:
        t3p = F.pad(t3, (pL, pR), mode=pad_mode_t)                     # [1,1,nt+pL+pR]
        padded_is_real = torch.ones(t3p.shape[-1], dtype=torch.bool, device=device)
        # non-constant pad ⇒ all entries are "real" (no invalid time positions)
    elif pad_mode_t in {"constant", "zero"}:
        t3p = F.pad(t3, (pL, pR), mode="constant", value=0.0)
        # track which positions came from real indices [0, nt-1]
        idx_line = torch.arange(-pL, nt + pR, device=device)
        padded_is_real = (idx_line >= 0) & (idx_line < nt)
    else:  # "none"
        t3p = t3  # no pad
        idx_line = torch.arange(0, nt, device=device)
        padded_is_real = torch.ones_like(idx_line, dtype=torch.bool)

    # Use 2D unfold with H=1 to make 1D sliding windows
    t4 = t3p.view(1, 1, 1, -1)                                         # [1,1,1,W]
    tw = F.unfold(t4, kernel_size=(1, kt), stride=(1, st))             # [1, kt, Nt]
    Nt = tw.shape[-1]
    times = tw.squeeze(0).transpose(0, 1).contiguous()                 # [Nt, kt]

    # Time-valid mask (only matters for constant/none)
    if pad_mode_t in {"reflect", "replicate", "circular"}:
        valid_t = torch.ones((Nt, kt), dtype=torch.bool, device=device)
    else:
        # build windows over the index line to know which positions are from real time
        idx4 = (idx_line.view(1, 1, 1, -1)).to(dtype=coords.dtype)
        iw = F.unfold(idx4, kernel_size=(1, kt), stride=(1, st)).to(torch.long)  # [1, kt, Nt]
        iw = iw.squeeze(0).transpose(0, 1)                                       # [Nt, kt]
        valid_t = ((iw >= 0) & (iw < nt))                                        # [Nt, kt]

    # Expand spatial patches across Nt windows and kt positions per window
    coords_xy = coords.unsqueeze(1).unsqueeze(3).repeat(1, Nt, 1, kt, 1)         # [L,Nt,P,kt,2]
    t_big     = times.view(1, Nt, 1, kt).repeat(L, 1, P, 1)                      # [L,Nt,P,kt]
    coords3   = torch.cat([coords_xy, t_big.unsqueeze(-1)], dim=-1)              # [L,Nt,P,kt,3]
    coords3   = coords3.view(L * Nt, P * kt, 3)

    # Build masks
    val_big = valid.unsqueeze(1).unsqueeze(3).repeat(1, Nt, 1, kt)               # [L,Nt,P,kt]
    bnd_big = bnd.unsqueeze(1).unsqueeze(3).repeat(1, Nt, 1, kt)                 # [L,Nt,P,kt]
    tm_big  = valid_t.view(1, Nt, 1, kt).repeat(L, 1, P, 1)                      # [L,Nt,P,kt]

    v_bool = (val_big > 0.5) & tm_big
    b_bool = (bnd_big > 0.5) & v_bool

    valid3 = v_bool.view(L * Nt, P * kt).to(dtype)
    bnd3   = b_bool.view(L * Nt, P * kt).to(dtype)

    # IC mask: near t0 within half-step tolerance
    tol   = 0.5 * (t1 - t0) / max(1, nt - 1)
    is_ic = (t_big - t0).abs() <= tol
    ic3   = (is_ic & v_bool).view(L * Nt, P * kt).to(dtype)

    meta_out = {**meta, "kt": kt, "st": st, "Nt": Nt, "P3": P * kt}
    return {"coords": coords3, "valid": valid3, "is_bnd": bnd3, "is_ic": ic3, "meta": meta_out}