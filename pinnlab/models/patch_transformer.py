import torch, torch.nn as nn
from .fourier_mlp import GaussianFF  # reuse your FF embed

class PatchTransformer(nn.Module):
    """
    Accepts:
      - patches: [B, S, d] tokens (d = 2 or 3: (x,y[,t]))
      - or flat points: [N, d] (for eval/inference)
    Optional inputs:
      - mask: [B, S] bool (True=valid) to ignore pads in attention & losses
      - token_type: [B, S] long in {0:int,1:bc,2:ic,3:pad} for a tiny type embedding
    """
    def __init__(self, cfg):
        super().__init__()
        d_in   = int(cfg["in_features"])
        d_out  = int(cfg["out_features"])
        d_model= int(cfg.get("d_model", 128))
        nhead  = int(cfg.get("n_heads", 4))
        depth  = int(cfg.get("depth", 4))
        ff_mult= int(cfg.get("ff_mult", 4))
        drop   = float(cfg.get("dropout", 0.0))

        ff_cfg = cfg.get("fourier_features", {"enabled": True, "m": 64, "sigma": 10.0})
        self.use_ff = bool(ff_cfg.get("enabled", True))
        if self.use_ff:
            m = int(ff_cfg.get("m", 64)); sigma = float(ff_cfg.get("sigma", 10.0))
            self.ff = GaussianFF(d_in, m, sigma)
            emb_in = 2*m
        else:
            self.ff = nn.Identity()
            emb_in = d_in

        self.input_proj = nn.Linear(emb_in, d_model)
        self.type_embed = nn.Embedding(4, d_model)  # 0:interior,1:bc,2:ic,3:pad
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*ff_mult, dropout=drop,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_out)
        )

    def forward(self, X, mask=None, token_type=None):
        # Normalize input shape
        was_flat = (X.dim()==2)
        if was_flat:
            X = X.unsqueeze(0)      # [1, N, d]
            if mask is not None and mask.dim()==1:
                mask = mask.unsqueeze(0)
            if token_type is not None and token_type.dim()==1:
                token_type = token_type.unsqueeze(0)

        Z = self.ff(X)
        Z = self.input_proj(Z)
        if token_type is not None:
            Z = Z + self.type_embed(token_type.clamp(0,3))

        # key_padding: True = IGNORE
        key_pad = None
        if mask is not None:
            key_pad = ~mask.bool()

        Z = self.encoder(Z, src_key_padding_mask=key_pad)
        Y = self.head(Z)  # [B, S, d_out]

        return Y.squeeze(0) if was_flat else Y