"""
test_patch_models.py - Quick test to verify the models work correctly
Run this to check if models can distinguish different inputs and learn.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import your models
from pinnlab.models.patch_ffn import PatchFFN
from pinnlab.models.patch_ffn_context import PatchFFNContext
from pinnlab.models.patch_cnn import PatchCNN
from pinnlab.models.patch_unet import UNetPatchCNN

def test_model_output_variance(model, patch_size=5, n_tests=10):
    """Test if model outputs different values for different inputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    P = patch_size * patch_size
    outputs = []
    
    with torch.no_grad():
        for i in range(n_tests):
            # Create different patches
            if i < n_tests // 2:
                # Bottom-left quadrant
                x = torch.rand(P, 2, device=device) * 0.5 - 1.0  # [-1, -0.5]
            else:
                # Top-right quadrant
                x = torch.rand(P, 2, device=device) * 0.5 + 0.5  # [0.5, 1]
            
            out = model(x).cpu().numpy().ravel()
            outputs.append(out)
    
    outputs = np.array(outputs)
    variance = np.var(outputs)
    mean_diff = np.abs(outputs[:n_tests//2].mean() - outputs[n_tests//2:].mean())
    
    print(f"Output variance: {variance:.6f}")
    print(f"Mean difference between quadrants: {mean_diff:.6f}")
    
    if variance < 1e-6:
        print("⚠️  WARNING: Model outputs nearly constant values!")
        return False
    else:
        print("✓ Model outputs show good variance")
        return True

def test_gradient_flow(model, patch_size=5):
    """Test if gradients flow through the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    P = patch_size * patch_size
    x = torch.randn(P, 2, device=device, requires_grad=True)
    
    # Forward pass
    out = model(x)
    loss = out.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = True
    small_gradients = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"⚠️  No gradient for {name}")
            has_gradients = False
        elif param.grad.abs().max().item() < 1e-8:
            small_gradients.append(name)
    
    if small_gradients:
        print(f"⚠️  Very small gradients in: {small_gradients[:3]}...")
    
    if has_gradients:
        print("✓ Gradients flow through all parameters")
    
    return has_gradients

def test_simple_fitting(model_class, config, patch_size=5, n_epochs=100):
    """Test if model can fit a simple function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    P = patch_size * patch_size
    
    # Create training data: u(x,y) = sin(πx) * sin(πy)
    def target_function(coords):
        x, y = coords[:, 0], coords[:, 1]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    
    losses = []
    
    for epoch in range(n_epochs):
        # Random patch
        coords = torch.rand(P, 2, device=device) * 2 - 1  # [-1, 1]
        coords.requires_grad = True
        
        # Forward
        pred = model(coords).squeeze(-1)
        target = target_function(coords)
        
        loss = nn.MSELoss()(pred, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # Check if loss decreased
    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    
    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    if final_loss < initial_loss * 0.5:
        print("✓ Model can learn!")
        return True
    else:
        print("⚠️  Model struggles to learn")
        return False

def main():
    print("=" * 60)
    print("Testing Patch-based Models")
    print("=" * 60)
    
    # Test configurations
    configs = {
        "PatchFFN": {
            "patch": {"x": 5, "y": 5},
            "in_features": 2,
            "out_features": 1,
            "hidden_dim": 128,
            "num_layers": 4,
            "activation": "tanh",
            "use_fourier_features": True,
            "fourier_scale": 1.0
        },
        "PatchFFNContext": {
            "patch": {"x": 5, "y": 5},
            "in_features": 2,
            "out_features": 1,
            "hidden_dim": 128,
            "num_layers": 4,
            "activation": "tanh",
            "context_dim": 64,
            "use_fourier_features": True,
            "fourier_scale": 1.0
        },
        "PatchCNN": {
            "patch": {"x": 8, "y": 8},  # CNN needs square patches
            "in_features": 2,
            "out_features": 1,
            "base_channels": 32,
            "num_blocks": 3,
            "activation": "gelu",
            "coord_channels": 8
        },
        "UNetPatchCNN": {
            "patch": {"x": 8, "y": 8},
            "in_features": 2,
            "out_features": 1,
            "base_channels": 32,
            "activation": "gelu",
            "coord_channels": 8
        }
    }
    
    models_to_test = [
        (PatchFFN, configs["PatchFFN"], 5),
        (PatchFFNContext, configs["PatchFFNContext"], 5),
        (PatchCNN, configs["PatchCNN"], 8),
        (UNetPatchCNN, configs["UNetPatchCNN"], 8)
    ]
    
    for model_class, config, patch_size in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_class.__name__}")
        print(f"{'='*60}")
        
        # Create model
        model = model_class(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test 1: Output variance
        print("\nTest 1: Output Variance")
        test_model_output_variance(model, patch_size)
        
        # Test 2: Gradient flow
        print("\nTest 2: Gradient Flow")
        test_gradient_flow(model, patch_size)
        
        # Test 3: Simple fitting
        print("\nTest 3: Simple Function Fitting")
        test_simple_fitting(model_class, config, patch_size, n_epochs=100)
        
        print()

if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)