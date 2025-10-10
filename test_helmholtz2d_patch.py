#!/usr/bin/env python
"""
Quick test script for time-dependent Helmholtz with patches
"""

import torch
import yaml
import sys
import os

# Add pinnlab to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_helmholtz2d_patch():
    """Test the Helmholtz2D patch implementation."""
    
    print("=" * 60)
    print("Testing Helmholtz2D Patch Implementation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load configs
    with open("configs/experiment/helmholtz2d_patch.yaml", "r") as f:
        exp_cfg = yaml.safe_load(f)
    
    with open("configs/model/patch_ffn_3d.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    # Ensure model config has correct patch dimensions from experiment
    model_cfg["patch"] = exp_cfg["patch"].copy()
    
    print(f"\nPatch configuration:")
    print(f"  x: {exp_cfg['patch']['x']}")
    print(f"  y: {exp_cfg['patch']['y']}")
    print(f"  t: {exp_cfg['patch']['t']}")
    print(f"  Total: {exp_cfg['patch']['x'] * exp_cfg['patch']['y'] * exp_cfg['patch']['t']}")
    
    # Initialize experiment
    from pinnlab.experiments.helmholtz2d_patch import Helmholtz2D_patch
    exp = Helmholtz2D_patch(exp_cfg, device)
    
    print("\n✓ Experiment initialized")
    
    # Initialize model
    from pinnlab.models.patch_ffn import PatchFFN
    model = PatchFFN(model_cfg).to(device)
    
    print(f"✓ Model initialized (expects {model.P} points per patch)")
    
    # Sample patches
    print("\nSampling patches...")
    batch = exp.sample_batch()
    patches = batch["patches"]
    
    print(f"  Patches shape: {patches['coords'].shape}")
    print(f"  Valid mask shape: {patches['valid'].shape}")
    print(f"  Boundary mask shape: {patches['is_bnd'].shape}")
    print(f"  Initial mask shape: {patches['is_ic'].shape}")
    
    # Test forward pass on a single patch
    print("\nTesting forward pass...")
    test_coords = patches['coords'][:1]  # Take first patch
    print(f"  Input shape: {test_coords.shape}")
    
    with torch.no_grad():
        output = model(test_coords)
    print(f"  Output shape: {output.shape}")
    print("✓ Forward pass successful")
    
    # Test loss computation (small batch)
    print("\nTesting loss computation...")
    small_batch = {
        "patches": {
            "coords": patches['coords'][:4],
            "valid": patches['valid'][:4],
            "is_bnd": patches['is_bnd'][:4],
            "is_ic": patches['is_ic'][:4]
        }
    }
    
    try:
        # Test PDE residual
        loss_pde = exp.pde_residual_loss(model, small_batch)
        print(f"  PDE loss: {loss_pde.item():.6f}")
        
        # Test boundary loss
        loss_bc = exp.boundary_loss(model, small_batch)
        print(f"  Boundary loss: {loss_bc.item():.6f}")
        
        # Test initial condition loss
        loss_ic = exp.initial_loss(model, small_batch)
        print(f"  Initial loss: {loss_ic.item():.6f}")
        
        print("✓ Loss computation successful")
    except Exception as e:
        print(f"✗ Error in loss computation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test evaluation
    print("\nTesting evaluation...")
    eval_cfg = {"nx": 20, "ny": 20}
    
    try:
        rel_l2 = exp.relative_l2_on_grid(model, eval_cfg)
        print(f"  Relative L2 error: {rel_l2:.6f}")
        print("✓ Evaluation successful")
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run the full training with:")
    print("python pinnlab/train.py \\")
    print("    --model_name patch_ffn \\")
    print("    --experiment_name helmholtz2d_patch \\")
    print("    --common_config configs/common_config.yaml \\")
    print("    --model_config configs/model/patch_ffn_3d.yaml \\")
    print("    --exp_config configs/experiment/helmholtz2d_patch.yaml")
    
    return True


if __name__ == "__main__":
    success = test_helmholtz2d_patch()
    sys.exit(0 if success else 1)