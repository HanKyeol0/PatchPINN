import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_act

class RelPosBias(nn.Module):
    """MLP mapping (dx,dy) -> scalar bias for attention logits."""
    def __init__(self, hidden:int=32, act:str="relu", dropout:float=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), get_act(act),
            nn.Linear(hidden, hidden), get_act(act),
            nn.Linear(hidden, 1),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        # rel: [P,P,2] -> [P,P]
        return self.drop(self.net(rel).squeeze(-1))

class MHSA(nn.Module):
    """Multi-head self-attention with optional relative positional bias."""
    def __init__(self, d_model:int, heads:int, attn_dropout:float=0.0,
                 use_relpos_bias:bool=True, relpos_hidden:int=32, act:str="relu"):
        super().__init__()
        assert d_model % heads == 0
        self.h, self.dh = heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(self.dh)
        self.use_rpb = use_relpos_bias
        self.rpb = RelPosBias(relpos_hidden, act=act) if use_relpos_bias else None

    def forward(self, x: torch.Tensor, rel: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [P,D], rel: [P,P,2]
        P, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)           # [P,D] ×3
        q = q.view(P, self.h, self.dh).transpose(0,1)    # [h,P,dh]
        k = k.view(P, self.h, self.dh).transpose(0,1)
        v = v.view(P, self.h, self.dh).transpose(0,1)
        logits = torch.einsum('hpd,hqd->hpq', q, k) * self.scale  # [h,P,P]
        if self.use_rpb:
            logits = logits + self.rpb(rel).unsqueeze(0)          # add shared bias
        attn = self.attn_drop(torch.softmax(logits, dim=-1))
        out = torch.einsum('hpq,hqd->hpd', attn, v).transpose(0,1).contiguous().view(P, D)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, heads:int, mlp_hidden:int,
                 act:str="relu", attn_dropout:float=0.0, resid_dropout:float=0.0,
                 prenorm:bool=True, use_relpos_bias:bool=True, relpos_hidden:int=32):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MHSA(d_model, heads, attn_dropout, use_relpos_bias, relpos_hidden, act=act)
        self.mlp   = nn.Sequential(nn.Linear(d_model, mlp_hidden), get_act(act), nn.Linear(mlp_hidden, d_model))
        self.drop1 = nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity()
        self.drop2 = nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rel: Optional[torch.Tensor]) -> torch.Tensor:
        if self.prenorm:
            x = x + self.drop1(self.attn(self.norm1(x), rel))
            x = x + self.drop2(self.mlp(self.norm2(x)))
            return x
        x = self.norm1(x + self.drop1(self.attn(x, rel)))
        x = self.norm2(x + self.drop2(self.mlp(x)))
        return x

class PatchAttention(nn.Module):
    """
    Fixed-size patch model.
      Input : X ∈ R^{P×in_features} (points-as-tokens, e.g., in_features=2 for (x,y))
      Output: û ∈ R^{P×out_features} (usually out_features=1)
    P must equal (patch.x+1)*(patch.y+1).
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        print("self.px, self.py:", self.px, self.py)
        assert self.px > 0 and self.py > 0, "Provide model.patch.x and model.patch.y (>0)."
        self.P  = (self.px + 1) * (self.py + 1)

        in_f   = int(cfg.get("in_features", 2))
        out_f  = int(cfg.get("out_features", 1))
        d_model = int(cfg.get("hidden_dim", 128))
        layers  = int(cfg.get("num_layers", 4))
        heads   = int(cfg.get("num_heads", 4))
        ff_mult = int(cfg.get("ff_mult", 4))
        act     = cfg.get("activation", "relu")
        attn_dropout  = float(cfg.get("attn_dropout", 0.0))
        resid_dropout = float(cfg.get("resid_dropout", 0.0))
        prenorm = bool(cfg.get("prenorm", True))
        use_rpb = bool(cfg.get("use_relpos_bias", True))
        rpb_hidden = int(cfg.get("relpos_hidden", 32))
        self.normalize_patch_coords = bool(cfg.get("normalize_patch_coords", True))

        self.embed = nn.Linear(in_f, d_model)
        mlp_hidden = ff_mult * d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, mlp_hidden, act, attn_dropout, resid_dropout,
                             prenorm, use_rpb, rpb_hidden)
            for _ in range(layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_f)

        self.apply(self._init)

    @torch.no_grad()
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def _pairwise_rel(self, X: torch.Tensor) -> torch.Tensor:
        # X: [P,2] -> rel: [P,P,2], normalized to ~[-1,1] if enabled
        rel = X[:, None, :] - X[None, :, :]
        if self.normalize_patch_coords:
            span = (X.max(0).values - X.min(0).values).clamp_min(1e-6)  # [2]
            rel = rel / span
        return rel

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.dim() == 2, f"Expected [P,in_features], got {tuple(X.shape)}"
        P = X.shape[0]
        assert P == self.P, f"Expected P={(self.P)} points, got P={P}."
        rel = self._pairwise_rel(X)  # [P,P,2]
        z = self.embed(X)
        for blk in self.blocks:
            z = blk(z, rel)
        return self.head(self.norm_out(z))  # [P,out_features]
