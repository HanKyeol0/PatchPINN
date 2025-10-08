import torch
import torch.nn as nn
from typing import Dict, Any
from .activation import get_act

class PatchFFNHardBC(nn.Module):
    """
    FFN for patch-based PINNs with hard boundary constraints.
    The output is modified to exactly satisfy boundary conditions.
    
    For domain [-1, 1] × [-1, 1] with u=0 on boundary:
    u(x,y) = (1-x²)(1-y²) * N(x,y)
    where N is the network output.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        # Patch configuration
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        self.P = self.px * self.py
        
        # Domain bounds (assuming [-1, 1] × [-1, 1])
        self.x_min = cfg.get("x_min", -1.0)
        self.x_max = cfg.get("x_max", 1.0)
        self.y_min = cfg.get("y_min", -1.0)
        self.y_max = cfg.get("y_max", 1.0)
        
        # Network configuration
        in_features = cfg.get("in_features", 2)
        out_features = cfg.get("out_features", 1)
        hidden_dim = cfg.get("hidden_dim", 128)
        num_layers = cfg.get("num_layers", 6)
        activation = cfg.get("activation", "tanh")
        
        # Build network
        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(get_act(activation))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_act(activation))
        
        layers.append(nn.Linear(hidden_dim, out_features))
        
        self.network = nn.Sequential(*layers)
        
        # Boundary condition function (can be learned or fixed)
        self.bc_type = cfg.get("bc_type", "zero")  # "zero" or "learned"
        if self.bc_type == "learned":
            # Learn the boundary values
            self.bc_net = nn.Sequential(
                nn.Linear(in_features, hidden_dim // 2),
                get_act(activation),
                nn.Linear(hidden_dim // 2, out_features)
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Smaller initialization to prevent explosion with distance function
            nn.init.xavier_normal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def distance_function(self, X):
        """
        Compute distance function that is 0 on boundary and positive inside.
        For rectangular domain [-1,1]×[-1,1]: D(x,y) = (1-x²)(1-y²)
        """
        # Normalize coordinates to [-1, 1]
        x_norm = 2 * (X[..., 0] - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (X[..., 1] - self.y_min) / (self.y_max - self.y_min) - 1
        
        # Distance function
        dist = (1 - x_norm**2) * (1 - y_norm**2)
        
        return dist.unsqueeze(-1) if dist.dim() < X.dim() else dist
    
    def boundary_function(self, X):
        """
        Function that equals the desired BC on boundary.
        For zero BC, this returns 0.
        """
        if self.bc_type == "zero":
            return torch.zeros_like(X[..., 0:1])
        elif self.bc_type == "learned":
            return self.bc_net(X)
        else:
            # Can add other BC types here
            raise ValueError(f"Unknown BC type: {self.bc_type}")
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hard boundary constraints.
        u(x,y) = D(x,y) * N(x,y) + B(x,y)
        where D is distance function, N is network output, B is boundary function.
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, D = X.shape
        assert P == self.P, f"Expected {self.P} points, got {P}"
        
        # Flatten for processing
        X_flat = X.reshape(B * P, D)
        
        # Network output
        N = self.network(X_flat)
        N = N.reshape(B, P, -1)
        
        # Distance function
        dist = self.distance_function(X)
        
        # Boundary function (usually 0 for homogeneous BC)
        bc_val = self.boundary_function(X)
        
        # Combine: u = distance * network + boundary
        # This ensures u = boundary on the boundary (where distance = 0)
        out = dist * N + bc_val
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out


class SinActivation(nn.Module):
    """Sin activation function, often works well for PINNs."""
    def forward(self, x):
        return torch.sin(x)


class PatchFFNSin(nn.Module):
    """
    FFN with sin activation - often prevents constant solutions.
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
        
        # Build network with sin activation
        self.first = nn.Linear(in_features, hidden_dim)
        
        self.hidden = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.last = nn.Linear(hidden_dim, out_features)
        
        # Initialize with specific scheme for sin networks
        self.apply(self._init_weights)
        
        # Special initialization for first layer (important for sin activation)
        with torch.no_grad():
            self.first.weight.uniform_(-1, 1)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            n = m.weight.shape[0]
            m.weight.data.uniform_(-np.sqrt(6/n), np.sqrt(6/n))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, D = X.shape
        assert P == self.P, f"Expected {self.P} points, got {P}"
        
        # Flatten
        x = X.reshape(B * P, D)
        
        # Forward with sin activation
        x = torch.sin(self.first(x))
        
        for layer in self.hidden:
            x = torch.sin(layer(x))
        
        x = self.last(x)
        
        # Reshape
        out = x.reshape(B, P, -1)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out