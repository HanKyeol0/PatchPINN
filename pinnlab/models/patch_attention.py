import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_act

class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for 2D coordinates."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D sinusoidal encoding"
        
    def forward(self, x):
        """
        Args:
            x: [B, N, 2] or [N, 2] tensor of 2D coordinates
        Returns:
            [B, N, dim] or [N, dim] tensor of sinusoidal features
        """
        device = x.device
        half_dim = self.dim // 4
        emb_x = math.log(10000) / (half_dim - 1)
        emb_x = torch.exp(torch.arange(half_dim, device=device) * -emb_x)
        emb_y = emb_x.clone()
        
        if x.dim() == 2:
            pos_x = x[:, 0:1] * emb_x  # [N, half_dim]
            pos_y = x[:, 1:2] * emb_y  # [N, half_dim]
        else:
            pos_x = x[:, :, 0:1] * emb_x  # [B, N, half_dim]
            pos_y = x[:, :, 1:2] * emb_y  # [B, N, half_dim]
        
        # Create sinusoidal features
        emb = torch.cat([
            torch.sin(pos_x), torch.cos(pos_x),
            torch.sin(pos_y), torch.cos(pos_y)
        ], dim=-1)
        
        return emb
    
class FourierFeatures(nn.Module):
    """Random Fourier features for positional encoding."""
    def __init__(self, in_features=2, out_features=256, scale=10.0):
        super().__init__()
        # Random matrix B for Fourier features (frozen)
        self.register_buffer('B', torch.randn(in_features, out_features // 2) * scale)
        
    def forward(self, x):
        """
        Args:
            x: [..., in_features] tensor
        Returns:
            [..., out_features] tensor of Fourier features
        """
        x_proj = x @ self.B  # [..., out_features // 2]
        return torch.cat([
            torch.sin(2 * math.pi * x_proj), 
            torch.cos(2 * math.pi * x_proj)
        ], dim=-1)
    
class LearnedPosEmbed(nn.Module):
    """Learnable positional embedding for patch points."""
    def __init__(self, max_patches=100, dim=128):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(max_patches, dim) * 0.02)
        
    def forward(self, n_points):
        """Return first n_points positional embeddings."""
        return self.embed[:n_points]

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
        # rel: [P,P,2] or [B,P,P,2] -> [P,P] or [B,P,P]
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

    def forward(self, x, rel): # x:[B,P,D], rel:[B,P,P,2]
        B, P, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # [B,P,D] ×3
        q = q.view(B, P, self.h, self.dh).transpose(1,2)  # [B,h,P,dh]
        k = k.view(B, P, self.h, self.dh).transpose(1,2)
        v = v.view(B, P, self.h, self.dh).transpose(1,2)
        
        logits = torch.einsum('bhpd,bhqd->bhpq', q, k) * self.scale  # [B,h,P,P]
        
        if self.use_rpb and rel is not None:
            bias = self.rpb(rel).unsqueeze(1)  # [B,1,P,P]
            logits = logits + bias
        
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhpq,bhqd->bhpd', attn, v).transpose(1,2).contiguous().view(B,P,D)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, heads:int, mlp_hidden:int,
                 act:str="relu", attn_dropout:float=0.0, resid_dropout:float=0.0,
                 prenorm:bool=True, use_relpos_bias:bool=True, relpos_hidden:int=32):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MHSA(d_model, heads, attn_dropout, use_relpos_bias, relpos_hidden, act=act)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), 
            get_act(act), 
            nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, d_model)
        )
        self.drop1 = nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity()
        self.drop2 = nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.prenorm:
            x = x + self.drop1(self.attn(self.norm1(x), rel))
            x = x + self.drop2(self.mlp(self.norm2(x)))
            return x
        x = self.norm1(x + self.drop1(self.attn(x, rel)))
        x = self.norm2(x + self.drop2(self.mlp(x)))
        return x

class PatchAttention(nn.Module):
    """
    Fixed-size patch model for PINNs.
    Input : X ∈ R^{P×in_features} where P = patch_x * patch_y
    Output: û ∈ R^{P×out_features} (usually out_features=1)
    
    patch_x and patch_y represent the number of points per patch in each dimension.
    For example, patch_x=3, patch_y=3 means 3×3=9 points per patch.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))  # points in x per patch
        self.py = int(patch.get("y"))  # points in y per patch
        print(f"Patch size: px={self.px}, py={self.py}")
        assert self.px > 0 and self.py > 0, "Provide model.patch.x and model.patch.y (>0)."
        
        # Total points per patch
        self.P = self.px * self.py
        print(f"Points per patch: P={self.P}")

        in_f = int(cfg.get("in_features", 2))
        out_f = int(cfg.get("out_features", 1))
        d_model = int(cfg.get("hidden_dim", 128))
        layers = int(cfg.get("num_layers", 4))
        heads = int(cfg.get("num_heads", 4))
        ff_mult = int(cfg.get("ff_mult", 4))
        act = cfg.get("activation", "relu")
        attn_dropout = float(cfg.get("attn_dropout", 0.0))
        resid_dropout = float(cfg.get("resid_dropout", 0.0))
        prenorm = bool(cfg.get("prenorm", True))
        use_rpb = bool(cfg.get("use_relpos_bias", True))
        rpb_hidden = int(cfg.get("relpos_hidden", 32))
        self.normalize_patch_coords = bool(cfg.get("normalize_patch_coords", True))

        # Model layers
        pos_encoding = cfg.get("pos_encoding", "fourier")  # Options: "fourier", "sinusoidal", "learned", "none"
        print(f"Using positional encoding: {pos_encoding}")
        
        if pos_encoding == "fourier":
            # Random Fourier features
            fourier_scale = cfg.get("fourier_scale", 5.0)
            self.pos_encoder = FourierFeatures(in_features=2, out_features=d_model, scale=fourier_scale)
            self.embed = nn.Linear(d_model, d_model)
        elif pos_encoding == "sinusoidal":
            # Sinusoidal positional encoding
            self.pos_encoder = SinusoidalPosEmbed(dim=d_model)
            self.embed = nn.Linear(d_model, d_model)
        elif pos_encoding == "learned":
            # Learned positional embeddings
            self.pos_encoder = LearnedPosEmbed(max_patches=self.P, dim=d_model)
            self.embed = nn.Linear(in_f, d_model)
            self.use_learned = True
        else:
            # No special positional encoding (original)
            self.pos_encoder = None
            self.embed = nn.Linear(in_f, d_model)

        mlp_hidden = ff_mult * d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, mlp_hidden, act, attn_dropout, resid_dropout,
                           prenorm, use_rpb, rpb_hidden)
            for _ in range(layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            get_act(act),
            nn.Linear(d_model // 2, out_f)
        )

        self.apply(self._init)

    @torch.no_grad()
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: 
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Tensor of shape [P, in_features] or [B, P, in_features]
               where P = patch_x * patch_y
        Returns:
            Tensor of shape [P, out_features] or [B, P, out_features]
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)  # [1,P,in_features]
        
        B, P, Din = X.shape
        assert P == self.P, f"Expected P={self.P}, got {P}"
        
        # ============================================
        # Apply positional encoding
        # ============================================
        if self.pos_encoder is not None:
            if isinstance(self.pos_encoder, LearnedPosEmbed):
                # Learned embeddings: add to coordinate embeddings
                z = self.embed(X)  # [B, P, d_model]
                pos_emb = self.pos_encoder(P).unsqueeze(0)  # [1, P, d_model]
                z = z + pos_emb
            else:
                # Fourier or Sinusoidal: encode coordinates directly
                X_spatial = X[:, :, :2]  # Just (x,y) coordinates
                z = self.pos_encoder(X_spatial)  # [B, P, d_model]
                z = self.embed(z)  # Project to model dimension
        else:
            # No positional encoding (original)
            z = self.embed(X)
        
        # Compute relative positions for attention bias (if used)
        if self.blocks[0].attn.use_rpb:
            rel = X[:, :, None, :2] - X[:, None, :, :2]  # [B, P, P, 2]
            if self.normalize_patch_coords:
                span = (X[:, :, :2].max(dim=1).values - X[:, :, :2].min(dim=1).values).clamp_min(1e-6)
                rel = rel / span[:, None, None, :]
        else:
            rel = None
        
        # Forward through transformer blocks
        for blk in self.blocks:
            z = blk(z, rel)
        
        # Output
        z = self.head(self.norm_out(z))  # [B, P, out_features]
        return z if z.shape[0] > 1 else z[0]