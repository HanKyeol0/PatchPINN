#!/usr/bin/env python
"""
Debug script to verify patch sizes and model expectations
"""

import yaml
import torch
import sys
import os

# Add pinnlab to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_patch_config():
    """Check if patch configurations are consistent."""
    
    print("=" * 60)
    print("PATCH SIZE CONFIGURATION CHECK")
    print("=" * 60)
    
    # Load experiment config
    try:
        with open("configs/experiment/helmholtz2d_patch.yaml", "r") as f:
            exp_cfg = yaml.safe_load(f)
        
        exp_px = exp_cfg["patch"]["x"]
        exp_py = exp_cfg["patch"]["y"]
        exp_pt = exp_cfg["patch"]["t"]
        
        print(f"\nExperiment config (helmholtz2d_patch.yaml):")
        print(f"  patch.x = {exp_px}")
        print(f"  patch.y = {exp_py}")
        print(f"  patch.t = {exp_pt}")
        print(f"  Total points per patch = {exp_px} × {exp_py} × {exp_pt} = {exp_px * exp_py * exp_pt}")
    except Exception as e:
        print(f"Error loading experiment config: {e}")
        return
    
    # Load model config
    try:
        with open("configs/model/patch_ffn.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)
        
        model_px = model_cfg["patch"]["x"]
        model_py = model_cfg["patch"]["y"]
        model_pt = model_cfg["patch"]["t"]
        
        print(f"\nModel config (patch_ffn.yaml):")
        print(f"  patch.x = {model_px}")
        print(f"  patch.y = {model_py}")
        print(f"  patch.t = {model_pt}")
        print(f"  Total points per patch = {model_px} × {model_py} × {model_pt} = {model_px * model_py * model_pt}")
    except Exception as e:
        print(f"Error loading model config: {e}")
        return
    
    # Check consistency
    print("\n" + "=" * 60)
    if exp_px == model_px and exp_py == model_py and exp_pt == model_pt:
        print("✓ CONFIGURATIONS MATCH!")
    else:
        print("✗ CONFIGURATION MISMATCH!")
        print("\nPlease ensure both configs have the same patch dimensions.")
        print("The model expects exactly the patch size specified in its config.")
    
    # Test model initialization
    print("\n" + "=" * 60)
    print("TESTING MODEL INITIALIZATION")
    print("=" * 60)
    
    try:
        from pinnlab.models.patch_ffn import PatchFFN
        
        # Override patch config from experiment
        model_cfg["patch"]["x"] = exp_px
        model_cfg["patch"]["y"] = exp_py
        model_cfg["patch"]["t"] = exp_pt
        
        model = PatchFFN(model_cfg)
        
        print(f"Model initialized successfully!")
        print(f"Model expects {model.P} points per patch")
        
        # Test forward pass
        test_batch_size = 2
        test_input = torch.randn(test_batch_size, model.P, 3)  # [B, P, 3]
        
        print(f"\nTesting forward pass with input shape: {test_input.shape}")
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful!")
        
    except Exception as e:
        print(f"✗ Error during model test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test patch extraction
    print("\n" + "=" * 60)
    print("TESTING PATCH EXTRACTION")
    print("=" * 60)
    
    try:
        from pinnlab.data.patches import extract_xy_patches, attach_time
        
        device = torch.device("cpu")
        
        # Test spatial patches
        spatial_patches = extract_xy_patches(
            xa=-1.0, xb=1.0, ya=-1.0, yb=1.0,
            nx=exp_cfg["grid"]["x"], ny=exp_cfg["grid"]["y"],
            kx=exp_px, ky=exp_py,
            sx=exp_cfg["stride"]["x"], sy=exp_cfg["stride"]["y"],
            pad_mode=exp_cfg["pad_mode"]["xy"],
            device=device
        )
        
        print(f"Spatial patches extracted:")
        print(f"  coords shape: {spatial_patches['coords'].shape}")
        print(f"  Number of spatial patches: {spatial_patches['coords'].shape[0]}")
        
        # Test time attachment
        patches_3d = attach_time(
            spatial_patches,
            t0=0.0, t1=1.0, nt=exp_cfg["grid"]["t"],
            kt=exp_pt, st=exp_cfg["stride"]["t"],
            pad_mode_t=exp_cfg["pad_mode"]["t"],
            sample_mode="sliding"
        )
        
        print(f"\n3D patches (with time):")
        print(f"  coords shape: {patches_3d['coords'].shape}")
        print(f"  Points per patch: {patches_3d['coords'].shape[1]}")
        print(f"  Expected points: {exp_px * exp_py * exp_pt}")
        
        if patches_3d['coords'].shape[1] == exp_px * exp_py * exp_pt:
            print("✓ Patch extraction produces correct size!")
        else:
            print("✗ Patch size mismatch in extraction!")
            
    except Exception as e:
        print(f"✗ Error during patch extraction test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Expected patch size: {exp_px} × {exp_py} × {exp_pt} = {exp_px * exp_py * exp_pt} points")
    print("\nIf you're seeing size mismatches, check:")
    print("1. Both config files have matching patch dimensions")
    print("2. The model config has the 't' dimension specified")
    print("3. The experiment properly creates 3D patches (x, y, t)")


if __name__ == "__main__":
    check_patch_config()