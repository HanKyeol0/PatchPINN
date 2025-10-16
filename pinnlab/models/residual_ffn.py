# pinnlab/models/residual_network.py
import math
from typing import Any, Dict

import torch
import torch.nn as nn
from typing import Iterable, Optional

# pinnlab/models/residual_ffn.py
import math
import torch
import torch.nn as nn


# ---- helpers ----
class Sine(nn.Module):
    def forward(self, x): 
        return torch.sin(x)

def get_act(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "sine": return Sine()
    if name == "swish" or name == "silu": return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")

class FlattenLast(nn.Module):
    """Apply a module to the last dimension only; keep leading dims intact."""
    def __init__(self, mod: nn.Module): 
        super().__init__(); self.mod = mod
    def forward(self, x):
        orig = x.shape
        if x.numel() == 0:
            return self.mod(x)
        y = x.view(-1, orig[-1])
        y = self.mod(y)
        return y.view(*orig[:-1], y.shape[-1])

# ---- Residual Block ----
class ResidualBlock(nn.Module):
    """
    Residual MLP block with a *linear projection* residual branch.

    y = P(x) + F(x)
      - P: single Linear (width -> width)
      - F: [Linear -> Act -> (Dropout?)] x L, staying at `width`
    """
    def __init__(
        self, width: int, 
        layers: int = 2,
        act: str = "tanh",
        dropout: float = 0.0,
        use_layernorm: bool = False,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.width = width
        self.residual_scale = float(residual_scale)

        blocks = []
        for i in range(layers):
            blocks.append(nn.Linear(width, width))
            blocks.append(get_act(act))
            if dropout and dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            if use_layernorm:
                blocks.append(nn.LayerNorm(width))
        self.main = nn.Sequential(*blocks) if blocks else nn.Identity()

        # Linear projection for residual path
        self.proj = nn.Linear(width, width, bias=True)

        # Kaiming init for ReLU-like, Xavier for others (simple heuristic)
        for m in self.main:
            if isinstance(m, nn.Linear):
                if "relu" in act or "silu" in act or "gelu" in act:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                else:
                    nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        return self.proj(x) + self.residual_scale * self.main(x)

# ---- Model ----
class ResidualFFN(nn.Module):
    """
    Flexible residual MLP for PatchPINN.

    Accepts both 2D (x,y) and 3D (x,y,t) inputs. Works on arbitrary leading dims:
      - [N, D] or [B, P, D] -> same leading -> last dim out_features

    Config (dict):
      in_features: int
      out_features: int
      hidden: int
      depth: int                 # number of residual blocks
      layers_per_block: int      # main-path layers inside each block
      activation: str
      dropout: float
      use_layernorm: bool
      residual_scale: float      # scales main path before addition
      input_stem: int            # number of Linear+Act before blocks (>=0)
      output_layers: int         # number of Linear+Act after blocks (>=0)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.in_features  = int(cfg.get("in_features", 2))
        self.out_features = int(cfg.get("out_features", 1))
        self.width        = int(cfg.get("hidden", 128))
        self.depth        = int(cfg.get("depth", 8))
        self.layers_per_block = int(cfg.get("layers_per_block", 2))
        self.activation   = str(cfg.get("activation", "tanh"))
        self.dropout      = float(cfg.get("dropout", 0.0))
        self.use_ln       = bool(cfg.get("use_layernorm", False))
        self.res_scale    = float(cfg.get("residual_scale", 1.0))
        self.input_stem_n = int(cfg.get("input_stem", 1))
        self.output_layers_n = int(cfg.get("output_layers", 0))

        # Input stem: project to width
        stem = []
        last = self.in_features
        for i in range(max(0, self.input_stem_n - 1)):
            stem += [nn.Linear(last, self.width), get_act(self.activation)]
            if self.dropout > 0: stem.append(nn.Dropout(self.dropout))
            if self.use_ln: stem.append(nn.LayerNorm(self.width))
            last = self.width
        # final to width
        stem += [nn.Linear(last, self.width)]
        self.stem = nn.Sequential(*stem)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(
                width=self.width,
                layers=self.layers_per_block,
                act=self.activation,
                dropout=self.dropout,
                use_layernorm=self.use_ln,
                residual_scale=self.res_scale,
            ) for _ in range(self.depth)
        ])

        # Optional post-MLP layers (width -> width)
        outs = []
        for i in range(self.output_layers_n):
            outs += [nn.Linear(self.width, self.width), get_act(self.activation)]
            if self.dropout > 0: outs.append(nn.Dropout(self.dropout))
            if self.use_ln: outs.append(nn.LayerNorm(self.width))
        self.post = nn.Sequential(*outs) if len(outs) else nn.Identity()

        # Head
        self.head = nn.Linear(self.width, self.out_features)

        # Wrap with FlattenLast so we preserve leading dims automatically
        self._f_stem  = FlattenLast(self.stem)
        self._f_post  = FlattenLast(self.post)
        self._f_head  = FlattenLast(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., D] with D in {self.in_features} (e.g., 2 for (x,y), 3 for (x,y,t))
        returns: [..., out_features]
        """
        # Just ensure last dim matches; do not enforce {2,3} to keep generic
        assert x.shape[-1] == self.in_features, \
            f"Expected input last-dim={self.in_features}, got {x.shape[-1]}"
        y = self._f_stem(x)
        for blk in self.blocks:
            y = FlattenLast(blk)(y)
        y = self._f_post(y)
        y = self._f_head(y)
        return y


# # Local activation (keeps model self-contained)
# def get_act(name: str) -> nn.Module:
#     name = str(name).lower()
#     if name == "tanh": return nn.Tanh()
#     if name == "relu": return nn.ReLU()
#     if name == "gelu": return nn.GELU()
#     if name == "sine":
#         class Sine(nn.Module):
#             def forward(self, x): return torch.sin(x)
#         return Sine()
#     raise ValueError(f"Unknown activation {name}")


# # ---- Optional Fourier features ----------------------------------------------
# class FourierFeatures(nn.Module):
#     """Random Fourier features for coordinate encoding."""
#     def __init__(self, in_features=2, out_features=128, scale=1.0):
#         super().__init__()
#         out_features = int(out_features)
#         if out_features % 2 == 1:
#             out_features += 1  # even for sin/cos pairing
#         self.out_features = out_features
#         B = torch.randn(in_features, out_features // 2) * float(scale)
#         self.register_buffer("B", B)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [..., in_features]
#         x_proj = x @ self.B  # [..., out_features//2]
#         return torch.cat([torch.sin(2 * torch.pi * x_proj),
#                           torch.cos(2 * torch.pi * x_proj)], dim=-1)


# # ---- Residual block with projection shortcut --------------------------------
# class ResidualBlock(nn.Module):
#     """
#     Residual block with projection shortcut:
#       y = Main(x) + Linear(x)
#     Main path can be 2 or 3 Linear layers with activations (and optional dropout).
#     """
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         hidden_dim: int,
#         activation: str = "tanh",
#         layers_per_block: int = 2,   # 2 or 3
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         assert layers_per_block in (2, 3), "layers_per_block must be 2 or 3"
#         self.drop_p = float(dropout)

#         def Act(): return get_act(activation)

#         if layers_per_block == 2:
#             self.main = nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim),
#                 Act(),
#                 nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity(),
#                 nn.Linear(hidden_dim, out_dim),
#             )
#         else:
#             self.main = nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim),
#                 Act(),
#                 nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 Act(),
#                 nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity(),
#                 nn.Linear(hidden_dim, out_dim),
#             )

#         # Projection shortcut to match shapes exactly
#         self.shortcut = nn.Linear(in_dim, out_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.main(x) + self.shortcut(x)


# # ---- Model -------------------------------------------------------------------
# class ResidualFFN(nn.Module):
#     """
#     Patch-compatible residual MLP with projection shortcuts.

#     I/O (matches PatchFFN/PatchCNN):
#       - Accepts [P, D] or [B, P, D]
#       - Returns [P, C] or [B, P, C]

#     Robust to time-independent inputs (D=2) or time-dependent (D=3):
#       - If model expects 3 but gets 2, it pads a zero 't' (configurable).
#       - If model expects 2 but gets 3, it drops the last dim (configurable).
#     """
#     def __init__(self, cfg: Dict[str, Any]):
#         super().__init__()

#         # --- Core sizes (what the network is *built for*)
#         self.in_features  = int(cfg.get("in_features", 2))   # 2: (x,y), 3: (x,y,t)
#         self.out_features = int(cfg.get("out_features", 1))
#         hidden_dim        = int(cfg.get("hidden_dim", 128))
#         num_blocks        = int(cfg.get("num_blocks", 4)); assert num_blocks >= 2
#         layers_per_block  = int(cfg.get("layers_per_block", 2)); assert layers_per_block in (2, 3)

#         activation = str(cfg.get("activation", "tanh"))
#         dropout    = float(cfg.get("dropout", 0.0))

#         # --- Input-dimension alignment policies
#         self.auto_pad_missing_time = bool(cfg.get("auto_pad_missing_time", True))   # 2->3
#         self.auto_drop_extra_dim   = bool(cfg.get("auto_drop_extra_dim", True))     # 3->2
#         self.missing_time_value    = float(cfg.get("missing_time_value", 0.0))      # pad value

#         # --- Fourier features
#         self.use_fourier   = bool(cfg.get("use_fourier_features", True))
#         fourier_scale      = float(cfg.get("fourier_scale", 1.0))
#         fourier_dim        = int(cfg.get("fourier_dim", hidden_dim))

#         if self.use_fourier:
#             self.fourier = FourierFeatures(self.in_features, out_features=fourier_dim, scale=fourier_scale)
#             stem_in_dim = fourier_dim
#         else:
#             self.fourier = None
#             stem_in_dim = self.in_features

#         # Build block dims: [in -> hidden] + (num_blocks-2)*[hidden -> hidden] + [hidden -> out]
#         dims = []
#         dims.append((stem_in_dim, hidden_dim))              # first block
#         for _ in range(num_blocks - 2):
#             dims.append((hidden_dim, hidden_dim))           # middle blocks
#         dims.append((hidden_dim, self.out_features))        # final block

#         self.blocks = nn.ModuleList([
#             ResidualBlock(
#                 in_dim=d_in,
#                 out_dim=d_out,
#                 hidden_dim=hidden_dim,
#                 activation=activation,
#                 layers_per_block=layers_per_block,
#                 dropout=dropout,
#             )
#             for (d_in, d_out) in dims
#         ])

#         self._init_weights(activation)

#         # Friendly summary print (no dependency on 't' in cfg)
#         patch = cfg.get("patch", {}) or {}
#         px, py = int(patch.get("x", 0)), int(patch.get("y", 0))
#         P_hint = px * py if (px and py) else None
#         tag = f"P~{P_hint}" if P_hint else "P=?"
#         print(f"ResidualNetwork (projection-skip): {tag}, in={self.in_features}, out={self.out_features}, "
#               f"hidden={hidden_dim}, blocks={num_blocks}({layers_per_block}-layer), "
#               f"act={activation}, fourier={self.use_fourier}")

#     # ---- initialization helpers ---------------------------------------------
#     def _init_weights(self, activation: str):
#         act = activation.lower()
#         if act == "sine":
#             self._siren_init()
#             return

#         def init_lin(m):
#             if isinstance(m, nn.Linear):
#                 if act in ("relu", "gelu"):
#                     nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#                 else:
#                     nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#         self.apply(init_lin)

#         # Make the very last linear in the last block small for stability
#         last_lin = None
#         for m in reversed(self.blocks[-1].main):
#             if isinstance(m, nn.Linear):
#                 last_lin = m
#                 break
#         if last_lin is not None:
#             with torch.no_grad():
#                 last_lin.weight.mul_(0.1)

#     def _siren_init(self, w0: float = 30.0):
#         # SIREN-style init (good for sine activations)
#         def init_first(m: nn.Module, in_dim: int):
#             if isinstance(m, nn.Linear):
#                 bound = 1.0 / in_dim
#                 nn.init.uniform_(m.weight, -bound, bound)
#                 if m.bias is not None:
#                     nn.init.uniform_(m.bias, -bound, bound)

#         def init_sine(m: nn.Module):
#             if isinstance(m, nn.Linear):
#                 in_dim = m.weight.size(1)
#                 bound = math.sqrt(6 / in_dim) / w0
#                 nn.init.uniform_(m.weight, -bound, bound)
#                 if m.bias is not None:
#                     nn.init.uniform_(m.bias, -bound, bound)

#         # Init all linears in mains & shortcuts
#         modules = []
#         for blk in self.blocks:
#             modules.append(blk.shortcut)
#             for mm in blk.main:
#                 if isinstance(mm, nn.Linear):
#                     modules.append(mm)

#         # First two get "first-layer" scheme; others sine init
#         used_first = 0
#         for m in modules:
#             if isinstance(m, nn.Linear):
#                 if used_first < 2:
#                     init_first(m, m.in_features)
#                     used_first += 1
#                 else:
#                     init_sine(m)

#         # Small final output
#         last_lin = None
#         for m in reversed(self.blocks[-1].main):
#             if isinstance(m, nn.Linear):
#                 last_lin = m
#                 break
#         if last_lin is not None:
#             with torch.no_grad():
#                 last_lin.weight.mul_(0.1)
#                 if last_lin.bias is not None:
#                     last_lin.bias.zero_()

#     # ---- input alignment helpers --------------------------------------------
#     def _align_input_dim(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Aligns x to the model's expected input feature dim (self.in_features).
#         Policies:
#           - D+1 == expected and auto_pad_missing_time: pad a constant 't'
#           - D-1 == expected and auto_drop_extra_dim: drop last column (assumed 't')
#         """
#         D = x.shape[-1]
#         E = self.in_features
#         if D == E:
#             return x
#         if D + 1 == E and self.auto_pad_missing_time:
#             # pad missing time (append constant column)
#             pad = torch.full((x.size(0), 1), self.missing_time_value, dtype=x.dtype, device=x.device)
#             return torch.cat([x, pad], dim=-1)
#         if D - 1 == E and self.auto_drop_extra_dim:
#             # drop extra dim (assume last is time)
#             return x[:, :E]
#         raise ValueError(
#             f"[ResidualNetwork] Input feature dim mismatch: got D={D}, expected {E}. "
#             f"Consider setting in_features to {D}, or enable auto_pad_missing_time/auto_drop_extra_dim."
#         )

#     # ---- forward -------------------------------------------------------------
#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """
#         X: [P, D] or [B, P, D]
#         Returns: [P, C] or [B, P, C]
#         """
#         squeeze = False
#         if X.dim() == 2:
#             X = X.unsqueeze(0)  # [1, P, D]
#             squeeze = True

#         B, P, D = X.shape
#         x = X.reshape(B * P, D)  # [BP, D]

#         # Align to expected feature dim (handles missing/provided time)
#         x = self._align_input_dim(x)  # [BP, self.in_features]

#         # Optional Fourier features
#         if self.use_fourier:
#             x = self.fourier(x)  # [BP, fourier_dim]

#         # Residual stack
#         for blk in self.blocks:
#             x = blk(x)

#         out = x.view(B, P, -1)   # [-1] is out_features at the last block
#         if squeeze:
#             out = out.squeeze(0)
#         return out
