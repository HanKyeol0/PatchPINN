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
    # pad order for F.pad on 4D [N,C,H,W] is (W_left, W_right, H_top, H_bottom)
    (pw0, pw1), (ph0, ph1) = pad_hw
    if mode == "zero":
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
    img, mask, (xv, yv) = _build_xy_grid(xa, xb, ya, yb, nx, ny, device)
    # padding (so boundary points can appear centered if desired)
    ph0 = (ky - 1) // 2
    ph1 = ky - 1 - ph0
    pw0 = (kx - 1) // 2
    pw1 = kx - 1 - pw0
    pad_hw = ((pw0, pw1), (ph0, ph1))

    pad_mode_eff = {"constant": "zero"}.get(pad_mode, pad_mode)
    img_p  = _pad2d(img,  pad_hw, pad_mode_eff, value=0.0)
    if pad_mode_eff == "circular":
        mask_p = _pad2d(mask, pad_hw, "circular"); mask_p[:] = 1.0
    else:
        mask_p = _pad2d(mask, pad_hw, "zero", value=0.0)

    unfold = torch.nn.Unfold(kernel_size=(ky, kx), stride=(sy, sx))

    # Unfold coordinates: [1, 2*P, L] -> [L, 2*P]
    patches_xy = unfold(img_p).squeeze(0).transpose(0, 1).contiguous()
    P = kx * ky
    L = patches_xy.shape[0]
    # Split channels: first P are X, next P are Y, then stack to [L, P, 2]
    xs = patches_xy[:, :P]
    ys = patches_xy[:, P:]
    coords = torch.stack([xs, ys], dim=-1)  # [L, P, 2]

    # Unfold mask: [1, 1*P, L] -> [L, P] (valid cells)
    patches_mask = unfold(mask_p).squeeze(0).transpose(0, 1).contiguous()
    valid = (patches_mask > 0.5).to(coords.dtype)

    # Mark boundary points (spatial only)
    eps_x = (xb - xa) / max(1, nx - 1)
    eps_y = (yb - ya) / max(1, ny - 1)
    x = coords[..., 0]; y = coords[..., 1]
    is_bnd = ((x - xa).abs() <= 0.5*eps_x) | ((x - xb).abs() <= 0.5*eps_x) | \
             ((y - ya).abs() <= 0.5*eps_y) | ((y - yb).abs() <= 0.5*eps_y)
    is_bnd = (is_bnd & (valid > 0)).to(coords.dtype)

    meta = {
        "L": L, "P": P, "kx": kx, "ky": ky, "sx": sx, "sy": sy,
        "nx": nx, "ny": ny, "pad": (pw0, pw1, ph0, ph1)
    }
    return {"coords": coords, "valid": valid, "is_bnd": is_bnd, "meta": meta}

def attach_time(
    patches: Dict[str, torch.Tensor],
    t0: float, t1: float, nt: int,
    kt: int, st: int,
    pad_mode_t: str,
    sample_mode: str = "sliding",
) -> Dict[str, torch.Tensor]:
    """
    Expand 2D spatial patches into 3D (x,y,t) patches.

    Inputs (from extract_xy_patches):
      patches["coords"]: [L, P, 2]
      patches["valid"] : [L, P]
      patches["is_bnd"]: [L, P]  (spatial boundary)

    Outputs:
      coords: [L*Nt, P*kt, 3]
      valid : [L*Nt, P*kt]
      is_bnd: [L*Nt, P*kt]  (spatial boundary only)
      is_ic : [L*Nt, P*kt]  (t == t0)
    """
    coords = patches["coords"]     # [L,P,2]
    valid  = patches["valid"]      # [L,P]
    is_bnd = patches["is_bnd"]     # [L,P]
    device = coords.device

    ts = torch.linspace(t0, t1, nt, device=device)   # [nt]

    # Build 1D time signal as 4D so nn.Unfold can consume it
    t4  = ts.view(1, 1, 1, -1)                       # [1,1,1,nt]
    m4  = torch.ones_like(t4)                        # [1,1,1,nt]

    # Padding along time
    pL0 = (kt - 1) // 2
    pL1 = kt - 1 - pL0
    if pad_mode_t in ("reflect", "replicate", "circular"):
        t4p = F.pad(t4, (pL0, pL1), mode=pad_mode_t)
        m4p = F.pad(m4, (pL0, pL1), mode=pad_mode_t)
        if pad_mode_t == "circular":
            m4p[:] = 1.0
    elif pad_mode_t == "zero":
        t4p = F.pad(t4, (pL0, pL1), mode="constant", value=float("nan"))
        m4p = F.pad(m4, (pL0, pL1), mode="constant", value=0.0)
    elif pad_mode_t == "none":
        t4p, m4p = t4, m4
    else:
        raise ValueError(pad_mode_t)

    # Slide a window of length kt with stride st along time
    unfold1d = torch.nn.Unfold(kernel_size=(1, kt), stride=(1, st))
    tpatch = unfold1d(t4p).squeeze(0).T   # [Nt, kt]
    mpatch = unfold1d(m4p).squeeze(0).T   # [Nt, kt]
    Nt = tpatch.shape[0]

    L, P, _ = coords.shape

    # ---- build broadcastable tensors with matching ranks ----
    # (x,y) replicated across (Nt, kt)
    # coords: [L,P,2] -> [L, Nt, P, kt, 2]
    xy_big  = coords[:, None, :, None, :].repeat(1, Nt, 1, kt, 1)

    # valid & boundary masks: [L,P] -> [L, Nt, P, kt]
    val_big = valid[:,  None, :, None].repeat(1, Nt, 1, kt)
    bnd_big = is_bnd[:, None, :, None].repeat(1, Nt, 1, kt)

    # time windows: [Nt,kt] -> [L, Nt, P, kt]
    t_big   = tpatch[None, :, None, :].repeat(L, 1, P, 1)
    tm_big  = mpatch[None, :, None, :].repeat(L, 1, P, 1)

    # ---- concatenate along the last coord dim and flatten patches ----
    coords3 = torch.cat([xy_big[..., 0:1], xy_big[..., 1:2], t_big[..., None]], dim=-1)
    coords3 = coords3.reshape(L * Nt, P * kt, 3)

    # bool masks
    v_bool  = (val_big > 0.5) & (tm_big > 0.5)               # [L,Nt,P,kt] bool
    b_bool  = (bnd_big > 0.5) & v_bool                        # boundary & valid

    valid3  = v_bool.reshape(L * Nt, P * kt).to(coords3.dtype)
    bnd3    = b_bool.reshape(L * Nt, P * kt).to(coords3.dtype)

    # IC mask (t == t0) within half step tolerance
    tol = 0.5 * (t1 - t0) / max(1, nt - 1)
    is_ic = ((t_big - t0).abs() <= tol) & v_bool              # bool
    ic3   = is_ic.reshape(L * Nt, P * kt).to(coords3.dtype)

    meta = {**patches["meta"], "kt": kt, "st": st, "Nt": Nt, "P3": P * kt}
    return {"coords": coords3, "valid": valid3, "is_bnd": bnd3, "is_ic": ic3, "meta": meta}