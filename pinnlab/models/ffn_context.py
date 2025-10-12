import torch
import torch.nn as nn
from typing import Dict, Any
from .activation import get_act

class FourierFeatures(nn.Module):
    """Random Fourier features for better position encoding."""
    def __init__(self, in_features=2, out_features=128, scale=1.0):
        super().__init__()
        self.register_buffer('B', torch.randn(in_features, out_features // 2) * scale)
        
    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(2 * torch.pi * x_proj), 
                         torch.cos(2 * torch.pi * x_proj)], dim=-1)

class FFNContext(nn.Module):
    """
    Enhanced FFN that also considers patch-level context.
    Combines point-wise processing with patch-level aggregation.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        # Patch configuration
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        self.P = self.px * self.py
        
        # Network configuration
        in_features = cfg.get("in_features", 2)
        out_features = cfg.get("out_features", 1)
        hidden_dim = cfg.get("hidden_dim", 128)
        num_layers = cfg.get("num_layers", 6)
        activation = cfg.get("activation", "tanh")
        context_dim = cfg.get("context_dim", 64)
        
        # Fourier features for position encoding
        use_fourier = cfg.get("use_fourier_features", True)
        fourier_scale = cfg.get("fourier_scale", 1.0)
        
        if use_fourier:
            self.fourier = FourierFeatures(in_features, hidden_dim, fourier_scale)
            point_input_dim = hidden_dim
        else:
            self.fourier = None
            point_input_dim = in_features
        
        # Point-wise encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(point_input_dim, hidden_dim),
            get_act(activation),
            nn.Linear(hidden_dim, hidden_dim),
            get_act(activation),
        )
        
        # Patch context aggregator (processes entire patch)
        self.context_net = nn.Sequential(
            nn.Linear(self.P * hidden_dim, context_dim),
            get_act(activation),
            nn.Linear(context_dim, context_dim),
            get_act(activation),
        )
        
        # Final decoder (combines point features with context)
        decoder_layers = []
        decoder_input = hidden_dim + context_dim
        
        for i in range(num_layers - 2):
            if i == 0:
                decoder_layers.append(nn.Linear(decoder_input, hidden_dim))
            else:
                decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(get_act(activation))
        
        decoder_layers.append(nn.Linear(hidden_dim, out_features))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, D = X.shape
        
        # Apply Fourier features if available
        if self.fourier is not None:
            X_encoded = self.fourier(X.reshape(B * P, D))
            X_encoded = X_encoded.reshape(B, P, -1)
        else:
            X_encoded = X
        
        # Encode each point
        point_features = self.point_encoder(X_encoded.reshape(B * P, -1))
        point_features = point_features.reshape(B, P, -1)  # [B, P, hidden_dim]
        
        # Get patch-level context
        patch_flat = point_features.reshape(B, -1)  # [B, P * hidden_dim]
        context = self.context_net(patch_flat)  # [B, context_dim]
        
        # Broadcast context to all points
        context_expanded = context.unsqueeze(1).expand(-1, P, -1)  # [B, P, context_dim]
        
        # Combine point features with context
        combined = torch.cat([point_features, context_expanded], dim=-1)  # [B, P, hidden_dim + context_dim]
        
        # Decode to output
        out = self.decoder(combined.reshape(B * P, -1))
        out = out.reshape(B, P, -1)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out