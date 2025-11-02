import torch
import torch.nn as nn
from typing import Dict, Any
from .activation import get_act

class FourierFeatures(nn.Module):
    """Random Fourier features for better position encoding."""
    def __init__(self, in_features: int, out_features: int = 128, scale=1.0):
        super().__init__()
        self.register_buffer('B', torch.randn(in_features, out_features // 2) * scale)
        
    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(2 * torch.pi * x_proj), 
                         torch.cos(2 * torch.pi * x_proj)], dim=-1)

class FFN(nn.Module):
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
        self.px = int(patch.get("x", 3))  # Default to 3
        self.py = int(patch.get("y", 3))  # Default to 3
        
        # Determine if this is 2D or 3D problem based on input features
        self.in_features = cfg.get("in_features", 2)
        
        if self.in_features == 2:
            # 2D steady-state problem
            self.pt = 1
            print(f"FFN (2D): {self.px}x{self.py}")
        elif self.in_features == 3:
            # 3D time-dependent problem
            self.pt = int(patch.get("t", 3))  # Default to 3 for time dimension
            print(f"FFN (3D): {self.px}x{self.py}x{self.pt}")
        else:
            # General case
            self.pt = int(patch.get("t", 1))
            print(f"FFN: in_features={self.in_features}, patch=({self.px},{self.py},{self.pt})")
            
        self.P = self.px * self.py * self.pt
        print(f"[FFN] Points per patch P = {self.P}")
        
        # Network configuration
        out_features = cfg.get("out_features", 1)
        hidden_dim = cfg.get("hidden_dim", 128)
        num_layers = cfg.get("num_layers", 6)
        activation = get_act(cfg.get("activation", "tanh"))

        # Dropout
        self.dropout_p = float(cfg.get("dropout", 0.0))
        Drop = (lambda: nn.Dropout(self.dropout_p)) if self.dropout_p > 0.0 else (lambda: nn.Identity())
        if self.dropout_p > 0:
            print(f"[FFN] Using dropout p={self.dropout_p}")
        
        # Optional Fourier features
        use_fourier = cfg.get("use_fourier_features", True)
        fourier_scale = cfg.get("fourier_scale", 1.0)
        fourier_dim = int(cfg.get("fourier_dim", 64))  # default sensible width
        
        if use_fourier:
            self.fourier = FourierFeatures(self.in_features, out_features=fourier_dim, scale=fourier_scale)
            self.point_feat_dim = fourier_dim
        else:
            self.fourier = None
            self.point_feat_dim = self.in_features
            print(f"[FFN] Fourier features disabled: point_feat_dim={self.point_feat_dim}")
        
        # The FIRST layer must take the entire patch vector:
        self.input_dim = self.P * self.point_feat_dim
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(activation)

        # Optional: Layer normalization for stability
        self.use_layer_norm = cfg.get("use_layer_norm", False)
        if self.use_layer_norm:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
            ])
        
        # Hidden layers with optional residual connections
        self.use_residual = cfg.get("use_residual", False)
        if self.use_residual:
            self.residual_layers = nn.ModuleList()
            for i in range(num_layers - 2):
                layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    activation,
                    Drop()
                    *( [nn.LayerNorm(hidden_dim)] if self.use_layer_norm else [] )
                )
                self.residual_layers.append(layer)
            # Output layer
            self.output_layer = nn.Linear(hidden_dim, self.P * self.out_features)
        else:
            # Standard feedforward
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(Drop())
                if self.use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
            # Output layer
            layers.append(nn.Linear(hidden_dim, self.P * out_features))
            self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier initialization scaled for tanh/sigmoid
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, X: torch.Tensor, ep=None) -> torch.Tensor:
        """
        Args:
            X: [P, in_features] or [B, P, in_features]
               where P = px * py (for 2D) or px * py * pt (for 3D)
        Returns:
            [P, out_features] or [B, P, out_features]
        """
        # Handle both 2D and 3D inputs
        if X.dim() == 2:
            # [P, in_features] -> [1, P, in_features]
            if ep==0:
                print("X shape1:", X.shape)
            X = X.unsqueeze(0)
            if ep==0:
                print("X shape2:", X.shape)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, D = X.shape
        
        if ep==0:
            print("Batch size:", B)
            print("Points per patch:", P)
            print("Input feature dim:", D)

        # Flatten batch and points dimensions
        X_flat = X.reshape(B * P, D)  # [B*P, D]
        
        # Apply Fourier features if available
        if self.fourier is not None:
            X_flat = self.fourier(X_flat)
        
        # Forward through network
        if self.use_residual:
            # With residual connections
            out = X_flat
            # Input layer (already applied if not using residual)
            if not hasattr(self, 'network'):
                # Apply first layer
                out = self.residual_layers[0](out)
                # Apply residual layers
                for i in range(1, len(self.residual_layers)):
                    identity = out
                    out = self.residual_layers[i](out)
                    if i % 2 == 1:  # Add residual every 2 layers
                        out = out + identity
                # Output layer
                out = self.output_layer(out)
        else:
            # Standard feedforward
            out = self.network(X_flat)  # [B*P, out_features]
        
        # Apply layer norm if enabled
        if self.use_layer_norm and hasattr(self, 'norm_layers'):
            # This would need to be integrated into the forward pass above
            pass
        
        # Reshape back
        out = out.reshape(B, P, -1)  # [B, P, out_features]
        
        if squeeze_output:
            out = out.squeeze(0)  # [P, out_features]
        
        return out