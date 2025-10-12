# pinnlab/models/residual_network.py
import math
from typing import Any, Dict
import torch
import torch.nn as nn
from .activation import get_act


# --- Optional Fourier features (same spirit as PatchFFN) ----------------------
class FourierFeatures(nn.Module):
    """Random Fourier features for coordinate encoding."""
    def __init__(self, in_features=2, out_features=128, scale=1.0):
        super().__init__()
        # out_features should be even; we use half for sin and half for cos
        out_features = int(out_features)
        if out_features % 2 == 1:
            out_features += 1
        self.out_features = out_features
        self.register_buffer("B", torch.randn(in_features, out_features // 2) * float(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        x_proj = x @ self.B  # [..., out_features//2]
        return torch.cat([torch.sin(2 * torch.pi * x_proj),
                          torch.cos(2 * torch.pi * x_proj)], dim=-1)  # [..., out_features]


# --- Residual block -----------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Two-layer MLP block with a residual (pre-activation style LN optional).
    Operates point-wise: input shape [N, D], returns [N, D].
    """
    def __init__(
        self,
        dim: int,
        *,
        activation: str = "tanh",
        dropout: float = 0.0,
        norm: str = "none",        # 'none' | 'layernorm'
        pre_norm: bool = True,
        res_scale: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.act = get_act(activation)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.res_scale = float(res_scale)

        if norm.lower() == "layernorm":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.pre_norm = bool(pre_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.pre_norm:
            y = self.norm1(y)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        if not self.pre_norm:
            y = self.norm1(y)

        if self.pre_norm:
            y = self.norm2(y)
        y = self.fc2(y)
        if not self.pre_norm:
            y = self.norm2(y)

        return x + self.res_scale * y


# --- Model --------------------------------------------------------------------
class ResidualFFN(nn.Module):
    """
    Residual MLP for Patch-PINN.

    Input/Output contract (matches PatchFFN / PatchCNN):
      - Accepts [P, D] or [B, P, D], where P = px*py (for 2D) or px*py*pt (for 3D).
      - Returns [P, C] or [B, P, C].

    Config keys (examples):
      in_features: int (default 2)
      out_features: int (default 1)
      hidden_dim: int (default 128)
      num_blocks: int  (residual blocks; default derived from num_layers)
      # or: num_layers: int (total linears like FFN; we convert to blocks)

      activation: str  ("tanh", "relu", "gelu", "sine", ...)
      dropout: float   (default 0.0)
      norm: "none"|"layernorm" (default "none")
      pre_norm: bool   (default True)
      res_scale: float (default 1.0)

      use_fourier_features: bool (default True)
      fourier_scale: float       (default 1.0)

      # Patch config (used to assert P)
      patch: { x: int, y: int, t: optional int }
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        patch = cfg.get("patch", {}) or {}
        self.px = int(patch.get("x", 3))
        self.py = int(patch.get("y", 3))
        in_features = int(cfg.get("in_features", 2))

        # Determine patch point count P (supports optional time dimension)
        if in_features >= 3:  # treat as (x,y,t) unless specified otherwise
            self.pt = int(patch.get("t", 1))
        else:
            self.pt = 1
        self.P = int(self.px * self.py * self.pt)

        # Network hyperparams
        self.out_features = int(cfg.get("out_features", 1))
        hidden = int(cfg.get("hidden_dim", 128))
        activation = str(cfg.get("activation", "tanh"))
        dropout = float(cfg.get("dropout", 0.0))
        norm = str(cfg.get("norm", "none")).lower()
        pre_norm = bool(cfg.get("pre_norm", True))
        res_scale = float(cfg.get("res_scale", 1.0))

        # Blocks count: prefer explicit num_blocks; else derive from num_layers like FFN
        if "num_blocks" in cfg:
            num_blocks = int(cfg["num_blocks"])
        else:
            # If num_layers is given as in FFN (input + (num_layers-2) hidden + output),
            # map to ~half that many residual blocks.
            nl = int(cfg.get("num_layers", 6))
            num_blocks = max(1, (nl - 2) // 2)

        # Optional Fourier features (use hidden as Fourier dim like PatchFFN)
        use_fourier = bool(cfg.get("use_fourier_features", True))
        fourier_scale = float(cfg.get("fourier_scale", 1.0))
        if use_fourier:
            self.fourier = FourierFeatures(in_features, out_features=hidden, scale=fourier_scale)
            input_dim = hidden
        else:
            self.fourier = None
            input_dim = in_features

        # Stem -> hidden
        self.stem = nn.Linear(input_dim, hidden)
        self.stem_act = get_act(activation)

        # Residual trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden,
                activation=activation,
                dropout=dropout,
                norm=norm,
                pre_norm=pre_norm,
                res_scale=res_scale,
            )
            for _ in range(num_blocks)
        ])

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden) if norm == "layernorm" else nn.Identity(),
            nn.Linear(hidden, self.out_features)
        )

        # Initialization
        self._init_weights(activation)

        # Log a quick summary line (helps sanity-check in your logs)
        if self.pt == 1:
            print(f"ResidualNetwork (2D): {self.px}x{self.py}={self.P} points | "
                  f"hidden={hidden}, blocks={num_blocks}, act={activation}, fourier={use_fourier}")
        else:
            print(f"ResidualNetwork (3D): {self.px}x{self.py}x{self.pt}={self.P} points | "
                  f"hidden={hidden}, blocks={num_blocks}, act={activation}, fourier={use_fourier}")

    # ---- init helpers --------------------------------------------------------
    def _init_weights(self, activation: str):
        act = activation.lower()
        if act == "sine":
            self._siren_init()
        else:
            def init_lin(m):
                if isinstance(m, nn.Linear):
                    if act in ("relu", "gelu"):
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    else:
                        nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.apply(init_lin)
            # small head for stability
            if isinstance(self.head[-1], nn.Linear):
                with torch.no_grad():
                    self.head[-1].weight.mul_(0.1)

    def _siren_init(self, w0: float = 30.0):
        # SIREN-style init for sine activations
        def init_first(m: nn.Module, in_dim: int):
            if isinstance(m, nn.Linear):
                bound = 1.0 / in_dim
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None: nn.init.uniform_(m.bias, -bound, bound)

        def init_sine(m: nn.Module):
            if isinstance(m, nn.Linear):
                in_dim = m.weight.size(1)
                bound = math.sqrt(6 / in_dim) / w0
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None: nn.init.uniform_(m.bias, -bound, bound)

        # Stem (first layer) uses special init
        if isinstance(self.stem, nn.Linear):
            init_first(self.stem, self.stem.in_features)
        # Residual block linears
        for b in self.blocks:
            init_sine(b.fc1)
            init_sine(b.fc2)
        # Head small
        if isinstance(self.head[-1], nn.Linear):
            with torch.no_grad():
                self.head[-1].weight.mul_(0.1)
                if self.head[-1].bias is not None:
                    self.head[-1].bias.zero_()

    # ---- forward -------------------------------------------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [P, D] or [B, P, D]
        Returns: [P, C] or [B, P, C]
        """
        squeeze = False
        if X.dim() == 2:
            X = X.unsqueeze(0)     # [1, P, D]
            squeeze = True

        B, P, D = X.shape

        # We allow small mismatch at eval (e.g., non-overlapping final tiles),
        # but warn if it's far off expected patch size.
        if P != self.P and abs(P - self.P) > self.P:
            print(f"[ResidualNetwork] Warning: expected P={self.P}, got {P}")

        x = X.reshape(B * P, D)  # [BP, D]

        if self.fourier is not None:
            x = self.fourier(x)  # [BP, hidden]

        x = self.stem(x)         # [BP, hidden]
        x = self.stem_act(x)

        for blk in self.blocks:
            x = blk(x)           # [BP, hidden]

        out = self.head(x)       # [BP, out_features]
        out = out.view(B, P, self.out_features)

        if squeeze:
            out = out.squeeze(0) # [P, out_features]
        return out
