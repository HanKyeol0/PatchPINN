import torch
import torch.nn as nn
import numpy as np

def test_cnn_forward():
    """Test if CNN can do forward pass without NaN."""
    print("="*60)
    print("Testing CNN Forward Pass")
    print("="*60)
    
    from pinnlab.models.patch_cnn import PatchCNN, SimplePatchCNN
    
    # Test configuration
    config = {
        "patch": {"x": 4, "y": 4},  # 4x4 = 16 points
        "in_features": 2,
        "out_features": 1,
        "base_channels": 16,  # Start small
        "num_blocks": 2,
        "activation": "tanh",  # More stable than ReLU
        "coord_channels": 4,
        "use_batchnorm": False,  # Disable for debugging
        "dropout": 0.0
    }
    
    # Create model
    model = PatchCNN(config)
    
    # Test input
    coords = torch.tensor([
        [-1.0, -1.0], [-0.5, -1.0], [0.0, -1.0], [0.5, -1.0],
        [-1.0, -0.5], [-0.5, -0.5], [0.0, -0.5], [0.5, -0.5],
        [-1.0,  0.0], [-0.5,  0.0], [0.0,  0.0], [0.5,  0.0],
        [-1.0,  0.5], [-0.5,  0.5], [0.0,  0.5], [0.5,  0.5]
    ], dtype=torch.float32)
    
    print(f"Input shape: {coords.shape}")
    print(f"Input range: [{coords.min():.2f}, {coords.max():.2f}]")
    
    # Forward pass
    with torch.no_grad():
        output = model(coords)
    
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[:4, 0].numpy()}")
    
    # Check for NaN
    if torch.isnan(output).any():
        print("❌ NaN detected in output!")
        
        # Debug layer by layer
        print("\nDebugging layer by layer...")
        
        # Check encoding
        B = 1
        P = 16
        X = coords.unsqueeze(0)  # [1, 16, 2]
        
        encoded = model.coord_encoder(X)
        print(f"After encoding: shape={encoded.shape}, "
              f"min={encoded.min():.4f}, max={encoded.max():.4f}, "
              f"has_nan={torch.isnan(encoded).any()}")
        
        # Check each block
        features = encoded
        for i, block in enumerate(model.blocks):
            features = block(features)
            print(f"After block {i}: shape={features.shape}, "
                  f"min={features.min():.4f}, max={features.max():.4f}, "
                  f"has_nan={torch.isnan(features).any()}")
        
        return False
    else:
        print("✓ No NaN in output")
        return True

def test_cnn_gradients():
    """Test if gradients flow through CNN."""
    print("\n" + "="*60)
    print("Testing CNN Gradient Flow")
    print("="*60)
    
    from pinnlab.models.patch_cnn import SimplePatchCNN
    
    # Use simple CNN for testing
    config = {
        "patch": {"x": 4, "y": 4},
        "in_features": 2,
        "out_features": 1
    }
    
    model = SimplePatchCNN(config)
    
    # Input with gradients
    coords = torch.randn(16, 2, requires_grad=True)
    
    # Forward
    output = model(coords)
    
    # Check forward pass
    if torch.isnan(output).any():
        print("❌ NaN in forward pass!")
        return False
    
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Compute PDE-like loss
    loss = output.sum()
    
    # Try to compute gradients w.r.t. input
    try:
        grad_input = torch.autograd.grad(
            loss, coords,
            create_graph=True,
            retain_graph=True
        )[0]
        
        print(f"✓ Gradient w.r.t. input: shape={grad_input.shape}, "
              f"norm={grad_input.norm():.6f}")
        
        # Check model parameter gradients
        model.zero_grad()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if np.isnan(grad_norm) or np.isinf(grad_norm):
                    print(f"❌ NaN/Inf gradient in {name}")
                    return False
                print(f"  {name}: grad_norm={grad_norm:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error computing gradients: {e}")
        return False

def test_cnn_with_pde():
    """Test CNN with actual PDE computation."""
    print("\n" + "="*60)
    print("Testing CNN with PDE Loss")
    print("="*60)
    
    from pinnlab.models.patch_cnn import SimplePatchCNN
    
    config = {
        "patch": {"x": 4, "y": 4},
        "in_features": 2,
        "out_features": 1
    }
    
    model = SimplePatchCNN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Training for 10 iterations...")
    
    for i in range(10):
        # Random patch
        coords = torch.randn(16, 2, requires_grad=True) * 0.5  # Small values
        
        # Forward
        u = model(coords)
        
        # Simple loss (just output magnitude for testing)
        loss = u.pow(2).mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN in gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  Iter {i}: NaN/Inf in {name}")
                    has_nan = True
        
        if has_nan:
            print("❌ NaN detected in gradients!")
            return False
        
        # Clip gradients for safety
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        print(f"  Iter {i}: loss={loss.item():.6f}")
    
    print("✓ CNN training without NaN")
    return True

def diagnose_nan_issue():
    """Diagnose common causes of NaN in CNN."""
    print("\n" + "="*60)
    print("Common NaN Causes in CNN")
    print("="*60)
    
    print("""
    1. **Coordinate Encoding Issues**:
       - Input coordinates might have extreme values
       - Polynomial features (x², y²) can explode
       - Division by zero in normalization
    
    2. **Weight Initialization**:
       - Weights too large → activation overflow
       - Use smaller initialization (gain=0.1)
    
    3. **Activation Functions**:
       - ReLU can cause dead neurons
       - Try tanh or GELU instead
    
    4. **BatchNorm Issues**:
       - With batch_size=1, BatchNorm can be unstable
       - Try without BatchNorm first
    
    5. **Learning Rate**:
       - Too high LR can cause explosion
       - Start with 1e-5 or 1e-6
    """)
    
    print("\nRecommended fixes:")
    print("1. Use SimplePatchCNN for testing")
    print("2. Normalize inputs to [-1, 1]")
    print("3. Use tanh activation")
    print("4. Disable BatchNorm initially")
    print("5. Use very small learning rate (1e-5)")
    print("6. Clip gradients")

if __name__ == "__main__":
    print("CNN NaN Debugging")
    print("="*60)
    
    # Run tests
    test1 = test_cnn_forward()
    test2 = test_cnn_gradients()
    test3 = test_cnn_with_pde()
    
    if not all([test1, test2, test3]):
        diagnose_nan_issue()
        
        print("\n" + "="*60)
        print("QUICK FIX:")
        print("="*60)
        print("""
# Use this configuration for CNN:
config = {
    "patch": {"x": 4, "y": 4},  # Smaller patches
    "base_channels": 16,        # Fewer channels
    "num_blocks": 2,             # Fewer blocks
    "activation": "tanh",        # Stable activation
    "use_batchnorm": False,      # No BatchNorm
    "dropout": 0.0,
    "coord_channels": 4          # Fewer encoding channels
}

# And use smaller learning rate:
optimizer = Adam(model.parameters(), lr=1e-5)
        """)
    else:
        print("\n✓ All tests passed! CNN should work.")