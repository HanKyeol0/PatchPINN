import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .activation import get_act
import math

class CoordinateEncoder(nn.Module):
    """Encodes (x,y) coordinates into a multi-channel representation."""
    def __init__(self, out_channels=8, normalize=True):
        super().__init__()
        self.out_channels = out_channels
        self.normalize = normalize
        
    def forward(self, coords):
        """
        Args:
            coords: [B, P, 2] where P = H*W
        Returns:
            [B, out_channels, H, W]
        """
        B, P, _ = coords.shape
        # Assume square patches
        H = W = int(P ** 0.5)
        assert H * W == P, f"Patch must be square, got {P} points"
        
        # Reshape coords to grid - FIXED ORDER
        coords_grid = coords.reshape(B, H, W, 2)  # [B, H, W, 2]
        x = coords_grid[..., 0]  # [B, H, W]
        y = coords_grid[..., 1]  # [B, H, W]
        
        # Normalize to [-1, 1] if needed (helps with stability)
        if self.normalize:
            x = 2.0 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1.0
            y = 2.0 * (y - y.min()) / (y.max() - y.min() + 1e-8) - 1.0
        
        # Create multiple channels with different encodings
        channels = []
        
        # Direct coordinates
        channels.append(x)
        channels.append(y)
        
        # Polynomial features (be careful with magnitude)
        channels.append(x ** 2)
        channels.append(y ** 2) 
        channels.append(x * y)
        
        # Sinusoidal features (bounded)
        channels.append(torch.sin(math.pi * x))
        channels.append(torch.cos(math.pi * x))
        channels.append(torch.sin(math.pi * y))
        
        # Add more if needed to reach out_channels
        while len(channels) < self.out_channels:
            channels.append(torch.zeros_like(x))
        
        # Stack and select first out_channels
        encoded = torch.stack(channels[:self.out_channels], dim=1)  # [B, C, H, W]
        
        return encoded


class PatchCNN(nn.Module):
    """
    Fixed CNN-based model for patch PINNs.
    Key fixes:
    1. Better initialization
    2. Proper normalization
    3. Skip connections to preserve information
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        # Patch configuration
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        self.P = self.px * self.py
        
        # CNN requires square patches
        assert self.px == self.py, f"CNN requires square patches, got {self.px}x{self.py}"
        self.patch_size = self.px
        
        print(f"PatchCNN: {self.px}x{self.py} grid = {self.P} points")
        
        # Network configuration
        out_features = cfg.get("out_features", 1)
        base_channels = cfg.get("base_channels", 32)
        num_blocks = cfg.get("num_blocks", 3)
        activation = cfg.get("activation", "gelu")
        use_batchnorm = cfg.get("use_batchnorm", True)
        dropout = cfg.get("dropout", 0.0)
        
        # Coordinate encoder
        coord_channels = cfg.get("coord_channels", 8)
        self.coord_encoder = CoordinateEncoder(coord_channels, normalize=True)
        
        # Build CNN blocks with residual connections
        self.blocks = nn.ModuleList()
        channels = [coord_channels] + [base_channels * (2**i) for i in range(num_blocks)]
        
        for i in range(num_blocks):
            block = ConvBlock(
                channels[i], 
                channels[i+1], 
                activation,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
                use_residual=(channels[i] == channels[i+1])  # Residual if same size
            )
            self.blocks.append(block)
        
        final_channels = channels[-1]
        
        # Output head - pixel-wise prediction
        self.output_conv = nn.Sequential(
            nn.Conv2d(final_channels, final_channels // 2, 1),  # 1x1 conv
            nn.BatchNorm2d(final_channels // 2) if use_batchnorm else nn.Identity(),
            get_act(activation),
            nn.Conv2d(final_channels // 2, out_features, 1)  # Final 1x1 conv
        )
        
        self.out_features = out_features
        
        # Initialize weights carefully
        self.apply(self._init_weights)
        
        # Special small initialization for output layer
        with torch.no_grad():
            self.output_conv[-1].weight.mul_(0.01)
            if self.output_conv[-1].bias is not None:
                self.output_conv[-1].bias.zero_()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # He initialization for ReLU/GELU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [P, 2] or [B, P, 2] coordinates
        Returns:
            [P, out_features] or [B, P, out_features]
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, D = X.shape
        assert P == self.P, f"Expected {self.P} points, got {P}"
        H = W = self.patch_size
        
        # Encode coordinates to multi-channel image
        encoded = self.coord_encoder(X)  # [B, C, H, W]
        
        # Check for NaN after encoding
        if torch.isnan(encoded).any():
            print("WARNING: NaN in coordinate encoding!")
            # Replace NaN with 0
            encoded = torch.nan_to_num(encoded, 0.0)
        
        # Apply convolutional blocks
        features = encoded
        for i, block in enumerate(self.blocks):
            features = block(features)
            
            # Debug: check for NaN/Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"WARNING: NaN/Inf after block {i}")
                features = torch.nan_to_num(features, 0.0)
        
        # Output convolution - per-pixel prediction
        out = self.output_conv(features)  # [B, out_features, H, W]
        
        # Reshape to [B, P, out_features]
        out = out.permute(0, 2, 3, 1).reshape(B, P, self.out_features)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out


class ConvBlock(nn.Module):
    """Convolutional block with optional residual connection."""
    def __init__(self, in_channels, out_channels, activation,
                 use_batchnorm=True, dropout=0.0, use_residual=False):
        super().__init__()
        
        self.use_residual = use_residual
        
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_act(activation))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)
        self.activation = get_act(activation)
        
        # Projection for residual if needed
        if use_residual and in_channels != out_channels:
            self.project = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.project = None
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        
        if self.use_residual:
            if self.project is not None:
                identity = self.project(identity)
            out = out + identity
        
        return self.activation(out)


class SimplePatchCNN(nn.Module):
    """
    Simplified CNN for debugging - minimal architecture.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        self.P = self.px * self.py
        
        assert self.px == self.py, "CNN requires square patches"
        
        # Very simple architecture
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)  # Input: 2 channels (x, y)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)  # Output: 1 channel
        
        # Small weight initialization
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.xavier_normal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, X):
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, _ = X.shape
        H = W = int(P ** 0.5)
        
        # Create simple 2-channel input (x and y coordinates)
        X_grid = X.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        # Normalize to [-1, 1]
        X_grid = 2.0 * (X_grid - X_grid.min()) / (X_grid.max() - X_grid.min() + 1e-8) - 1.0
        
        # Simple forward pass
        x = torch.tanh(self.conv1(X_grid))
        x = torch.tanh(self.conv2(x))
        x = self.conv3(x)  # [B, 1, H, W]
        
        # Reshape output
        out = x.permute(0, 2, 3, 1).reshape(B, P, 1)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out