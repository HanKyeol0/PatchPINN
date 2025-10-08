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
        x_proj = x @ self.B # 
        return torch.cat([torch.sin(2 * torch.pi * x_proj), 
                         torch.cos(2 * torch.pi * x_proj)], dim=-1)

class PatchFFN(nn.Module):
    """
    Feed-Forward Network for patch-based PINNs.
    Processes each point independently with shared weights (like standard PINN).
    
    Key idea: The patch structure is used for efficient sampling/batching,
    but the network itself treats each point independently.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        # Patch configuration
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        self.P = self.px * self.py
        print(f"PatchFFN: {self.px}x{self.py} = {self.P} points per patch")
        
        # Network configuration
        in_features = cfg.get("in_features", 2)
        out_features = cfg.get("out_features", 1)
        hidden_dim = cfg.get("hidden_dim", 128)
        num_layers = cfg.get("num_layers", 6)
        activation = cfg.get("activation", "tanh")
        
        # Optional Fourier features
        use_fourier = cfg.get("use_fourier_features", True)
        fourier_scale = cfg.get("fourier_scale", 1.0)
        
        if use_fourier:
            self.fourier = FourierFeatures(in_features, hidden_dim, fourier_scale)
            input_dim = hidden_dim
        else:
            self.fourier = None
            input_dim = in_features
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(get_act(activation))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_act(activation))
            
            # Optional: Add residual connections for deeper networks
            if cfg.get("use_residual", False) and _ % 2 == 1:
                # We'll handle residual connections in forward pass
                pass
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_features))
        
        self.network = nn.Sequential(*layers)
        
        # Optional: Layer normalization for stability
        self.use_layer_norm = cfg.get("use_layer_norm", False)
        if self.use_layer_norm:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
            ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier initialization scaled for tanh/sigmoid
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [P, in_features] or [B, P, in_features]
               where P = px * py (points per patch)
        Returns:
            [P, out_features] or [B, P, out_features]
        """
        # Handle both 2D and 3D inputs
        if X.dim() == 2:
            # [P, in_features] -> [1, P, in_features]
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, D = X.shape
        assert P == self.P, f"Expected {self.P} points, got {P}"
        
        # Flatten batch and points dimensions
        X_flat = X.reshape(B * P, D)  # [B*P, D]
        
        # Apply Fourier features if available
        if self.fourier is not None:
            X_flat = self.fourier(X_flat)
        
        # Forward through network
        # Each point is processed independently
        out = self.network(X_flat)  # [B*P, out_features]
        
        # Reshape back
        out = out.reshape(B, P, -1)  # [B, P, out_features]
        
        if squeeze_output:
            out = out.squeeze(0)  # [P, out_features]
        
        return out