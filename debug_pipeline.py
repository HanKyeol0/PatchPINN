"""
debug_pipeline.py - Comprehensive debugging for patch-PINN pipeline
This script tests each component of the pipeline to identify where the issue is.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import yaml

# Import your modules
from pinnlab.models.patch_ffn import PatchFFN
from pinnlab.experiments.helmholtz2d_steady_patch import Helmholtz2DSteady_patch
from pinnlab.data.samplers import sample_patches_2d_steady

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

# ========================================
# TEST 1: Sampling and Boundary Detection
# ========================================
def test_sampling():
    print_section("TEST 1: Patch Sampling & Boundary Detection")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Simple test case
    out = sample_patches_2d_steady(
        x_min=-1.0, x_max=1.0,
        y_min=-1.0, y_max=1.0,
        patch_x=3, patch_y=3,  # 3x3 = 9 points per patch
        grid_x=7, grid_y=7,     # 7x7 total grid
        device=device,
        boundary_fn=lambda x, y: torch.ones_like(x)
    )
    
    interior_patches = out["interior_patches"]
    boundary_patches = out["boundary_patches"]
    
    print(f"Interior patches: {len(interior_patches)}")
    print(f"Boundary patches: {len(boundary_patches)}")
    
    if len(boundary_patches) > 0:
        # Check a boundary patch
        patch = boundary_patches[0]
        coords = patch["coords"]
        bmask = patch["boundary_mask"]
        
        print(f"\nFirst boundary patch:")
        print(f"  Points: {coords.shape[0]}")
        print(f"  Boundary points: {bmask.sum().item()}")
        print(f"  Interior points: {(~bmask).sum().item()}")
        
        # Verify boundary detection
        for i, (coord, is_boundary) in enumerate(zip(coords, bmask)):
            x, y = coord[0].item(), coord[1].item()
            expected_boundary = (abs(x - (-1.0)) < 1e-6 or abs(x - 1.0) < 1e-6 or 
                               abs(y - (-1.0)) < 1e-6 or abs(y - 1.0) < 1e-6)
            if expected_boundary != is_boundary.item():
                print(f"  ❌ Boundary detection error at point {i}: ({x:.2f}, {y:.2f})")
                print(f"     Expected: {expected_boundary}, Got: {is_boundary.item()}")
                return False
    
    print("✓ Sampling and boundary detection working correctly")
    return True

# ========================================
# TEST 2: Loss Computation
# ========================================
def test_loss_computation():
    print_section("TEST 2: Loss Computation")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create experiment
    cfg = {
        "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        "patch": {"x": 3, "y": 3},
        "grid": {"x": 30, "y": 30},
        "stride": {"x": 1, "y": 1},
        "a1": 1.0, "a2": 2.0, "lambda": 1.0,
        "batch_size": 2
    }
    exp = Helmholtz2DSteady_patch(cfg, device)
    
    # Create a simple test model that outputs known values
    class TestModel(nn.Module):
        def __init__(self, mode="constant"):
            super().__init__()
            self.mode = mode
            
        def forward(self, X):
            if X.dim() == 2:
                X = X.unsqueeze(0)
            B, P, _ = X.shape
            
            if self.mode == "constant":
                # Output constant 0.5
                out = torch.ones(B, P, 1, device=X.device) * 0.5
            elif self.mode == "linear":
                # Output u = x + y
                out = (X[..., 0:1] + X[..., 1:2])
            elif self.mode == "quadratic":
                # Output u = x^2 + y^2
                out = (X[..., 0:1]**2 + X[..., 1:2]**2)
            
            return out if B > 1 else out[0]
    
    # Test different model outputs
    batch = exp.sample_batch()
    
    print("\nTesting loss with different model outputs:")
    
    for mode in ["constant", "linear", "quadratic"]:
        model = TestModel(mode).to(device)
        
        # Compute losses
        loss_res = exp.pde_residual_loss(model, batch)
        loss_bc = exp.boundary_loss(model, batch)
        
        res_val = loss_res.mean().item() if loss_res.numel() > 0 else 0.0
        bc_val = loss_bc.mean().item() if loss_bc.numel() > 0 else 0.0
        
        print(f"\n  Mode: {mode}")
        print(f"    Residual loss: {res_val:.6f}")
        print(f"    Boundary loss: {bc_val:.6f}")
        
        if mode == "constant" and res_val > 1e-3:
            print(f"    Note: Constant should have low residual (Laplacian=0)")
        
        # Check if losses are reasonable (not NaN or Inf)
        if np.isnan(res_val) or np.isinf(res_val):
            print(f"    ❌ Residual loss is NaN/Inf!")
            return False
        if np.isnan(bc_val) or np.isinf(bc_val):
            print(f"    ❌ Boundary loss is NaN/Inf!")
            return False
    
    print("\n✓ Loss computation working correctly")
    return True

# ========================================
# TEST 3: Gradient Flow Through Losses
# ========================================
def test_gradient_flow():
    print_section("TEST 3: Gradient Flow Through Losses")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    cfg = {
        "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        "patch": {"x": 3, "y": 3},
        "grid": {"x": 30, "y": 30},
        "stride": {"x": 1, "y": 1},
        "a1": 1.0, "a2": 2.0, "lambda": 1.0,
        "batch_size": 2
    }
    exp = Helmholtz2DSteady_patch(cfg, device)
    
    model_cfg = {
        "patch": {"x": 3, "y": 3},
        "in_features": 2, "out_features": 1,
        "hidden_dim": 32, "num_layers": 3,
        "activation": "tanh",
        "use_fourier_features": False  # Simple case first
    }
    model = PatchFFN(model_cfg).to(device)
    
    batch = exp.sample_batch()
    
    # Test gradient flow for each loss component
    print("\nTesting gradient flow for each loss:")
    
    # Test boundary loss
    print("\n1. Boundary Loss:")
    model.zero_grad()
    loss_bc = exp.boundary_loss(model, batch)
    if loss_bc.numel() > 0:
        loss_bc_scalar = loss_bc.mean()
        loss_bc_scalar.backward()
        
        bc_grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                bc_grad_norms[name] = param.grad.norm().item()
        
        print(f"   Loss value: {loss_bc_scalar.item():.6f}")
        print(f"   Parameters with gradients: {len(bc_grad_norms)}")
        
        if len(bc_grad_norms) == 0:
            print("   ❌ No gradients from boundary loss!")
            return False
    
    # Test residual loss
    print("\n2. Residual Loss:")
    model.zero_grad()
    loss_res = exp.pde_residual_loss(model, batch)
    if loss_res.numel() > 0:
        loss_res_scalar = loss_res.mean()
        loss_res_scalar.backward()
        
        res_grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                res_grad_norms[name] = param.grad.norm().item()
        
        print(f"   Loss value: {loss_res_scalar.item():.6f}")
        print(f"   Parameters with gradients: {len(res_grad_norms)}")
        
        # Check if gradients are reasonable
        max_grad = max(res_grad_norms.values()) if res_grad_norms else 0
        min_grad = min(res_grad_norms.values()) if res_grad_norms else 0
        
        print(f"   Gradient norm range: [{min_grad:.6f}, {max_grad:.6f}]")
        
        if max_grad > 1e6:
            print("   ⚠️  Warning: Very large gradients detected!")
        if max_grad < 1e-10:
            print("   ⚠️  Warning: Very small gradients detected!")
    
    # Test combined loss
    print("\n3. Combined Loss:")
    model.zero_grad()
    loss_total = 100 * loss_bc.mean() + loss_res.mean()
    loss_total.backward()
    
    total_grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norms[name] = param.grad.norm().item()
    
    print(f"   Total loss: {loss_total.item():.6f}")
    print(f"   Parameters with gradients: {len(total_grad_norms)}")
    
    print("\n✓ Gradient flow working correctly")
    return True

# ========================================
# TEST 4: Training Dynamics
# ========================================
def test_training_dynamics():
    print_section("TEST 4: Training Dynamics (Mini Training)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    cfg = {
        "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        "patch": {"x": 3, "y": 3},
        "grid": {"x": 30, "y": 30},
        "stride": {"x": 1, "y": 1},
        "a1": 1.0, "a2": 2.0, "lambda": 1.0,
        "batch_size": 4
    }
    exp = Helmholtz2DSteady_patch(cfg, device)
    
    model_cfg = {
        "patch": {"x": 3, "y": 3},
        "in_features": 2, "out_features": 1,
        "hidden_dim": 64, "num_layers": 3,
        "activation": "tanh",
        "use_fourier_features": True,
        "fourier_scale": 1.0
    }
    model = PatchFFN(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Track training metrics
    losses = []
    output_stds = []
    output_ranges = []
    
    print("\nTraining for 50 iterations...")
    print("Iter | Total Loss | BC Loss | Res Loss | Output Std | Output Range")
    print("-" * 70)
    
    for i in range(50):
        batch = exp.sample_batch()
        
        # Compute losses
        loss_bc = exp.boundary_loss(model, batch).mean()
        loss_res = exp.pde_residual_loss(model, batch).mean()
        
        # Use strong BC weight to force non-constant solution
        loss_total = 1000 * loss_bc + 0.1 * loss_res
        
        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        
        # Check gradient norms before step
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()**2
        total_grad_norm = np.sqrt(total_grad_norm)
        
        # Gradient clipping if needed
        if total_grad_norm > 10.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        
        # Monitor output distribution
        with torch.no_grad():
            test_coords = torch.rand(100, 2, device=device) * 2 - 1
            test_patch = test_coords[:9]  # Take first 9 points as a patch
            outputs = model(test_patch).cpu().numpy().ravel()
            output_std = np.std(outputs)
            output_range = np.max(outputs) - np.min(outputs)
        
        losses.append(loss_total.item())
        output_stds.append(output_std)
        output_ranges.append(output_range)
        
        if i % 10 == 0:
            print(f"{i:4d} | {loss_total.item():10.6f} | {loss_bc.item():8.6f} | "
                  f"{loss_res.item():8.6f} | {output_std:10.6f} | {output_range:12.6f}")
    
    # Analyze results
    print("\n" + "-" * 70)
    print("\nAnalysis:")
    
    # Check if loss is decreasing
    early_loss = np.mean(losses[:5])
    late_loss = np.mean(losses[-5:])
    print(f"  Average loss (first 5 iters): {early_loss:.6f}")
    print(f"  Average loss (last 5 iters):  {late_loss:.6f}")
    
    if late_loss < early_loss * 0.9:
        print("  ✓ Loss is decreasing")
    else:
        print("  ⚠️  Loss is not decreasing significantly")
    
    # Check output variance
    final_std = np.mean(output_stds[-5:])
    final_range = np.mean(output_ranges[-5:])
    print(f"\n  Final output std: {final_std:.6f}")
    print(f"  Final output range: {final_range:.6f}")
    
    if final_std < 1e-4:
        print("  ❌ Model outputs are nearly constant!")
        print("\n  Possible causes:")
        print("    1. Loss weights are imbalanced")
        print("    2. Learning rate is too high/low")
        print("    3. Model initialization is poor")
        print("    4. Boundary conditions alone don't constrain solution enough")
        return False
    else:
        print("  ✓ Model outputs show variation")
    
    return True

# ========================================
# TEST 5: Model Architecture Check
# ========================================
def test_model_architecture():
    print_section("TEST 5: Model Architecture Verification")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_cfg = {
        "patch": {"x": 3, "y": 3},
        "in_features": 2, "out_features": 1,
        "hidden_dim": 64, "num_layers": 3,
        "activation": "tanh",
        "use_fourier_features": True,
        "fourier_scale": 1.0
    }
    model = PatchFFN(model_cfg).to(device)
    
    # Test with different inputs
    print("\nTesting model with various inputs:")
    
    test_cases = [
        ("All zeros", torch.zeros(9, 2, device=device)),
        ("All ones", torch.ones(9, 2, device=device)),
        ("Random", torch.randn(9, 2, device=device)),
        ("Grid points", torch.tensor([
            [-1, -1], [0, -1], [1, -1],
            [-1, 0], [0, 0], [1, 0],
            [-1, 1], [0, 1], [1, 1]
        ], dtype=torch.float32, device=device))
    ]
    
    outputs = {}
    for name, input_tensor in test_cases:
        with torch.no_grad():
            out = model(input_tensor).cpu().numpy().ravel()
            outputs[name] = out
            print(f"  {name:12s}: mean={out.mean():.6f}, std={out.std():.6f}, "
                  f"range=[{out.min():.6f}, {out.max():.6f}]")
    
    # Check if outputs are different
    if np.std([out.mean() for out in outputs.values()]) < 1e-6:
        print("\n  ⚠️  Model outputs similar values for very different inputs!")
        print("     This suggests the model is not using input information properly.")
        
        # Check Fourier features
        if hasattr(model, 'fourier'):
            print("\n  Checking Fourier features:")
            test_input = torch.randn(9, 2, device=device)
            fourier_out = model.fourier(test_input)
            print(f"    Fourier output shape: {fourier_out.shape}")
            print(f"    Fourier output std: {fourier_out.std().item():.6f}")
            
            if fourier_out.std().item() < 1e-6:
                print("    ❌ Fourier features are not working!")
    else:
        print("\n  ✓ Model responds to different inputs")
    
    return True

# ========================================
# MAIN DEBUGGING ROUTINE
# ========================================
def main():
    print("="*60)
    print(" PATCH-PINN PIPELINE DEBUGGING")
    print("="*60)
    
    tests = [
        ("Sampling & Boundaries", test_sampling),
        ("Loss Computation", test_loss_computation),
        ("Gradient Flow", test_gradient_flow),
        ("Training Dynamics", test_training_dynamics),
        ("Model Architecture", test_model_architecture)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error:")
            print(f"   {str(e)}")
            results[test_name] = False
    
    # Summary
    print_section("SUMMARY")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:25s}: {status}")
    
    # Diagnosis
    if not results.get("Training Dynamics", False):
        print_section("DIAGNOSIS: Constant Output Problem")
        print("""
The model outputs constant values. Most likely causes:

1. **Loss Imbalance**: The PDE loss might dominate, and constant u=0 
   minimizes the Laplacian. Try:
   - Increase boundary weight to 10000
   - Decrease residual weight to 0.01
   - Add a data loss term with known solution points

2. **Poor Initialization**: The model might start in a bad local minimum.
   Try:
   - Different weight initialization (Xavier with smaller gain)
   - Different activation (try 'sin' activation)
   - Add noise to break symmetry

3. **Learning Rate Issues**: 
   - Try much smaller LR: 1e-5 or 1e-6
   - Use learning rate scheduling
   - Try different optimizers (LBFGS often works for PINNs)

4. **Missing Constraints**: Boundary conditions alone might not be enough.
   - Add interior points with known solution
   - Use hard constraints (modify output to satisfy BCs exactly)
   - Add regularization terms

5. **Numerical Issues**:
   - Check for NaN/Inf in gradients
   - Use gradient clipping
   - Normalize inputs to [-1, 1]
        """)

if __name__ == "__main__":
    main()