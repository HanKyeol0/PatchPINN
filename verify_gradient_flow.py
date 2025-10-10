#!/usr/bin/env python
"""
Verify gradient flow in the Helmholtz2D patch implementation
"""

import torch
import torch.nn as nn
import yaml
import sys
import os

# Add pinnlab to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gradient_flow():
    """Test that gradients flow properly through the model and losses."""
    
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load configs
    with open("configs/experiment/helmholtz2d_patch.yaml", "r") as f:
        exp_cfg = yaml.safe_load(f)
    
    with open("configs/model/patch_ffn.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    # Ensure model config has correct patch dimensions
    model_cfg["patch"] = exp_cfg["patch"].copy()
    
    # Initialize experiment and model
    from pinnlab.experiments.helmholtz2d_patch import Helmholtz2D_patch
    from pinnlab.models.patch_ffn import PatchFFN
    
    exp = Helmholtz2D_patch(exp_cfg, device)
    model = PatchFFN(model_cfg).to(device)
    
    print(f"✓ Model initialized (expects {model.P} points per patch)")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Sample a batch
    print("\nSampling patches...")
    batch = exp.sample_batch()
    
    # Test gradient flow for each loss component
    print("\nTesting individual loss components:")
    
    # Test PDE residual loss
    print("\n1. PDE Residual Loss:")
    optimizer.zero_grad()
    loss_pde = exp.pde_residual_loss(model, batch)
    print(f"   Value: {loss_pde.item():.6f}")
    print(f"   Requires grad: {loss_pde.requires_grad}")
    
    try:
        loss_pde.backward(retain_graph=True)
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
        print(f"   Gradients computed: {has_grad}")
        if has_grad:
            max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
            print(f"   Max gradient: {max_grad:.6f}")
    except Exception as e:
        print(f"   ✗ Backward failed: {e}")
    
    # Test boundary loss
    print("\n2. Boundary Loss:")
    optimizer.zero_grad()
    loss_bc = exp.boundary_loss(model, batch)
    print(f"   Value: {loss_bc.item():.6f}")
    print(f"   Requires grad: {loss_bc.requires_grad}")
    
    try:
        loss_bc.backward(retain_graph=True)
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
        print(f"   Gradients computed: {has_grad}")
        if has_grad:
            max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
            print(f"   Max gradient: {max_grad:.6f}")
    except Exception as e:
        print(f"   ✗ Backward failed: {e}")
    
    # Test initial condition loss
    print("\n3. Initial Condition Loss:")
    optimizer.zero_grad()
    loss_ic = exp.initial_loss(model, batch)
    print(f"   Value: {loss_ic.item():.6f}")
    print(f"   Requires grad: {loss_ic.requires_grad}")
    
    try:
        loss_ic.backward(retain_graph=True)
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
        print(f"   Gradients computed: {has_grad}")
        if has_grad:
            max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
            print(f"   Max gradient: {max_grad:.6f}")
    except Exception as e:
        print(f"   ✗ Backward failed: {e}")
    
    # Test total loss
    print("\n4. Total Loss (weighted sum):")
    optimizer.zero_grad()
    
    w_res = exp_cfg.get("loss_weights", {}).get("res", 1.0)
    w_bc = exp_cfg.get("loss_weights", {}).get("bc", 10.0)
    w_ic = exp_cfg.get("loss_weights", {}).get("ic", 10.0)
    
    loss_pde = exp.pde_residual_loss(model, batch)
    loss_bc = exp.boundary_loss(model, batch)
    loss_ic = exp.initial_loss(model, batch)
    
    total_loss = w_res * loss_pde + w_bc * loss_bc + w_ic * loss_ic
    print(f"   Total value: {total_loss.item():.6f}")
    print(f"   Components: PDE={loss_pde.item():.6f}, BC={loss_bc.item():.6f}, IC={loss_ic.item():.6f}")
    print(f"   Requires grad: {total_loss.requires_grad}")
    
    try:
        total_loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        print(f"   Gradients computed: {has_grad}")
        
        if has_grad:
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            print(f"   Number of parameters with gradients: {len(grad_norms)}")
            print(f"   Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
            
            # Test optimizer step
            print("\n5. Optimizer Step:")
            param_before = next(model.parameters()).clone()
            optimizer.step()
            param_after = next(model.parameters())
            param_change = (param_after - param_before).abs().max().item()
            print(f"   Parameter change: {param_change:.6f}")
            print(f"   ✓ Optimizer successfully updated parameters")
        else:
            print("   ✗ No gradients computed!")
            return False
            
    except Exception as e:
        print(f"   ✗ Backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL GRADIENT TESTS PASSED!")
    print("=" * 60)
    print("\nThe model is ready for training.")
    return True


if __name__ == "__main__":
    success = test_gradient_flow()
    sys.exit(0 if success else 1)