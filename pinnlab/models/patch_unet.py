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


class UNetPatchCNN(nn.Module):
    """
    U-Net style CNN for patch-based PINNs.
    Better at preserving spatial information through skip connections.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        # Patch configuration
        patch = cfg.get("patch", {})
        self.px = int(patch.get("x"))
        self.py = int(patch.get("y"))
        self.P = self.px * self.py
        
        assert self.px == self.py, "U-Net requires square patches"
        self.patch_size = self.px
        
        # Network configuration
        out_features = cfg.get("out_features", 1)
        base_channels = cfg.get("base_channels", 32)
        activation = cfg.get("activation", "gelu")
        
        # Coordinate encoder
        coord_channels = cfg.get("coord_channels", 8)
        self.coord_encoder = CoordinateEncoder(coord_channels)
        
        # Encoder path
        self.enc1 = self._double_conv(coord_channels, base_channels, activation)
        self.enc2 = self._double_conv(base_channels, base_channels * 2, activation)
        self.enc3 = self._double_conv(base_channels * 2, base_channels * 4, activation)
        
        # Decoder path with skip connections
        self.dec2 = self._double_conv(base_channels * 6, base_channels * 2, activation)
        self.dec1 = self._double_conv(base_channels * 3, base_channels, activation)
        
        # Output layer (1x1 conv)
        self.output = nn.Conv2d(base_channels, out_features, 1)
        
        self.apply(self._init_weights)
    
    def _double_conv(self, in_channels, out_channels, activation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            get_act(activation),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            get_act(activation)
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, D = X.shape
        H = W = int(P ** 0.5)
        
        # Encode coordinates
        x = self.coord_encoder(X)  # [B, C, H, W]
        
        # Encoder
        enc1 = self.enc1(x)  # [B, 32, H, W]
        enc2 = self.enc2(enc1)  # [B, 64, H, W]
        enc3 = self.enc3(enc2)  # [B, 128, H, W]
        
        # Decoder with skip connections
        dec2 = self.dec2(torch.cat([enc3, enc2], dim=1))  # [B, 64, H, W]
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))  # [B, 32, H, W]
        
        # Output
        out = self.output(dec1)  # [B, out_features, H, W]
        
        # Reshape to [B, P, out_features]
        out = out.permute(0, 2, 3, 1).reshape(B, P, -1)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out