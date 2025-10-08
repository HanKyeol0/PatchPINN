"""
patch_cnn.py - Convolutional network for patch-based PINNs
Treats the patch as a 2D image and uses convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .activation import get_act

class CoordinateEncoder(nn.Module):
    """Encodes (x,y) coordinates into a multi-channel representation."""
    def __init__(self, out_channels=8):
        super().__init__()
        self.out_channels = out_channels
        
    def forward(self, coords):
        """
        Args:
            coords: [B, P, 2] where P = H*W
        Returns:
            [B, out_channels, H, W]
        """
        B, P, _ = coords.shape
        # Assume square patches for simplicity
        H = W = int(P ** 0.5)
        assert H * W == P, f"Patch must be square, got {P} points"
        
        # Reshape coords to grid
        x = coords[:, :, 0].reshape(B, H, W)  # [B, H, W]
        y = coords[:, :, 1].reshape(B, H, W)  # [B, H, W]
        
        # Create multiple channels with different encodings
        channels = []
        
        # Direct coordinates
        channels.append(x)
        channels.append(y)
        
        # Polynomial features
        channels.append(x ** 2)
        channels.append(y ** 2)
        channels.append(x * y)
        
        # Sinusoidal features for periodicity
        channels.append(torch.sin(torch.pi * x))
        channels.append(torch.cos(torch.pi * x))
        channels.append(torch.sin(torch.pi * y))
        
        # Stack first out_channels features
        encoded = torch.stack(channels[:self.out_channels], dim=1)  # [B, C, H, W]
        return encoded


class PatchCNN(nn.Module):
    """
    CNN-based model for patch PINNs.
    Treats each patch as a small image and applies convolutions.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        # Patch configuration
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        self.P = self.px * self.py
        
        # For CNN, patches should be square
        assert self.px == self.py, "CNN model requires square patches"
        self.patch_size = self.px
        
        print(f"PatchCNN: {self.px}x{self.py} grid")
        
        # Network configuration
        in_features = cfg.get("in_features", 2)
        out_features = cfg.get("out_features", 1)
        base_channels = cfg.get("base_channels", 32)
        num_blocks = cfg.get("num_blocks", 3)
        activation = cfg.get("activation", "gelu")
        
        # Coordinate encoder
        coord_channels = cfg.get("coord_channels", 8)
        self.coord_encoder = CoordinateEncoder(coord_channels)
        
        # Build CNN layers
        channels = [coord_channels]
        for i in range(num_blocks):
            channels.append(base_channels * (2 ** i))
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = self._make_block(
                channels[i], 
                channels[i + 1], 
                activation,
                use_residual=(i > 0)  # Skip first block for residual
            )
            self.blocks.append(block)
        
        # Global pooling and final layers
        final_channels = channels[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        
        # Decoder that outputs for each point
        self.decoder = nn.Sequential(
            nn.Linear(final_channels, final_channels * 2),
            get_act(activation),
            nn.Linear(final_channels * 2, final_channels),
            get_act(activation),
            nn.Linear(final_channels, self.P * out_features)
        )
        
        self.out_features = out_features
        self.apply(self._init_weights)
    
    def _make_block(self, in_channels, out_channels, activation, use_residual=False):
        """Create a convolutional block with optional residual connection."""
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_act(activation))
        
        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        block = nn.Sequential(*layers)
        
        # Add residual connection if needed
        if use_residual and in_channels == out_channels:
            return ResidualBlock(block)
        else:
            return block
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
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
        
        # Encode coordinates to multi-channel image
        encoded = self.coord_encoder(X)  # [B, C, H, W]
        
        # Apply convolutional blocks
        features = encoded
        for block in self.blocks:
            features = block(features)
        
        # Global pooling
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # [B, final_channels]
        
        # Decode to output for each point
        out = self.decoder(pooled)  # [B, P * out_features]
        out = out.reshape(B, P, self.out_features)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out


class ResidualBlock(nn.Module):
    """Residual wrapper for blocks."""
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        return x + self.block(x)