import torch
import torch.nn as nn
from .mlp import get_act
from .fourier_mlp import GaussianFF

class PatchAttention(nn.Module):
    """
    Transformer-style PINN that consumes a *patch* of space-time coordinates
    and predicts solution values per point.

    Inputs:
      - forward_patches(coords, valid):
          coords: [B, P, D]   (D = 2 for (x,y) or 3 for (x,y,t))
          valid : [B, P] in {0,1}  (mask for padded points; 1=valid, 0=pad)

      - forward(x): accepts [N, D] (pointwise fallback; *no* patch context)

    Config (model_config.yaml):
      in_features: 2|3
      out_features: 1
      d_model: 128
      nhead: 8
      num_layers: 4
      ff_multiplier: 4.0
      dropout: 0.0
      activation: tanh|relu|sine
      fourier:
        enabled: true|false
        m: 32
        sigma: 1.0
        concat_input: true|false
      head_hidden: d_model
    """
    def __init__(self, cfg):
        super().__init__()
        in_f  = int(cfg.get("in_features", 2))
        out_f = int(cfg.get("out_features", 1))
        d_model = int(cfg.get("d_model", 128))
        nhead   = int(cfg.get("nhead", 8))
        nlayer  = int(cfg.get("num_layers", 4))
        ff_mult = float(cfg.get("ff_multiplier", 4.0))
        dropout = float(cfg.get("dropout", 0.0))
        act_name = cfg.get("activation", "tanh")

        fcfg = cfg.get("fourier", {}) or {}
        use_ff = bool(fcfg.get("enabled", False))
        if use_ff:
            m     = int(fcfg.get("m", 32))
            sigma = float(fcfg.get("sigma", 1.0))
            self.ff = GaussianFF(in_f, m, sigma)
            self.concat_input = bool(fcfg.get("concat_input", True))
            in_embed = (in_f + 2*m) if self.concat_input else (2*m)
        else:
            self.ff = None
            self.concat_input = True
            in_embed = in_f

        self.input_proj = nn.Linear(in_embed, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(ff_mult * d_model),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayer)
        self.norm = nn.LayerNorm(d_model)

        hidden = int(cfg.get("head_hidden", d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden),
            get_act(act_name),
            nn.Linear(hidden, out_f),
        )

        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def _make_key_padding_mask(self, valid):
        """
        valid: [B,P] float/bool with 1 for valid, 0 for pad
        returns bool mask with True at PAD positions
        """
        if valid is None:
            return None
        if valid.dtype != torch.bool:
            return (valid < 0.5)
        return (~valid)

    def forward_patches(self, coords: torch.Tensor, valid: torch.Tensor = None):
        """
        coords: [B,P,D], valid: [B,P] (1=valid, 0=pad)
        returns: [B,P,1]
        """
        if coords.dim() != 3:
            raise ValueError(f"coords must be [B,P,D], got {list(coords.shape)}")

        B, P, D = coords.shape
        if self.ff is not None:
            z = self.ff(coords.reshape(-1, D))
            if self.concat_input:
                z = torch.cat([coords.reshape(-1, D), z], dim=-1)
        else:
            z = coords.reshape(-1, D)

        z = self.input_proj(z).reshape(B, P, -1)

        key_padding_mask = self._make_key_padding_mask(valid)  # [B,P] True means pad
        h = self.encoder(z, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        out = self.head(h)  # [B,P,1]
        return out

    def forward(self, x: torch.Tensor):
        """
        Fallback for [N,D] inputs (no attention context across patches).
        Returns [N,1]. For patch training, always call forward_patches(...).
        """
        if x.dim() == 2:
            y = self.forward_patches(x.unsqueeze(0), None).squeeze(0)
            return y
        elif x.dim() == 3:
            return self.forward_patches(x, None)
        else:
            raise ValueError("Input must be [N,D] or [B,P,D]")