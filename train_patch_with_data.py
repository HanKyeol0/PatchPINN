"""
train_patch_with_data.py - Modified training with data points to break constant solution
This approach adds known solution points to force the model to learn non-constant outputs.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import yaml
from tqdm import tqdm

def train_with_data_points(args):
    """Modified training that includes data points to prevent constant solutions."""
    
    # Load configs
    with open(args.common_config, 'r') as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_cfg = yaml.safe_load(f)
    with open(args.exp_config, 'r') as f:
        exp_cfg = yaml.safe_load(f)
    
    device = torch.device(base_cfg['device'] if torch.cuda.is_available() else 'cpu')
    
    # Import after configs are loaded
    from pinnlab.registry import get_model, get_experiment
    from pinnlab.utils.seed import seed_everything
    
    seed_everything(base_cfg['seed'])
    
    # Create model and experiment
    model = get_model(args.model_name)(model_cfg).to(device)
    exp = get_experiment(args.experiment_name)(exp_cfg, device)
    
    # CRITICAL: Create data points with known solution
    n_data = 100
    x_data = torch.rand(n_data, device=device) * 2 - 1  # [-1, 1]
    y_data = torch.rand(n_data, device=device) * 2 - 1
    
    # Compute exact solution at these points
    import math
    a1, a2 = exp_cfg.get('a1', 1.0), exp_cfg.get('a2', 2.0)
    u_exact = torch.sin(a1 * math.pi * x_data) * torch.sin(a2 * math.pi * y_data)
    
    # Group into patches for the model
    patch_size = model_cfg['patch']['x'] * model_cfg['patch']['y']
    data_patches = []
    data_targets = []
    
    for i in range(0, n_data - patch_size + 1, patch_size):
        coords = torch.stack([x_data[i:i+patch_size], y_data[i:i+patch_size]], dim=1)
        targets = u_exact[i:i+patch_size]
        data_patches.append(coords)
        data_targets.append(targets)
    
    print(f"Created {len(data_patches)} data patches with known solutions")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=base_cfg['train']['optimizer']['lr'])
    
    # Training with curriculum learning
    epochs = base_cfg['train']['epochs']
    
    print("\n" + "="*60)
    print("PHASE 1: Learning from Data Points (Force Non-Constant)")
    print("="*60)
    
    # Phase 1: Train only on data points first
    for epoch in range(min(500, epochs//4)):
        total_loss = 0
        
        for coords, targets in zip(data_patches, data_targets):
            pred = model(coords).squeeze(-1)
            loss = nn.MSELoss()(pred, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            # Check output variance
            with torch.no_grad():
                test_patch = data_patches[0]
                test_out = model(test_patch).cpu().numpy().ravel()
                output_std = np.std(test_out)
                
            print(f"Epoch {epoch:4d}: Data Loss = {total_loss/len(data_patches):.6f}, "
                  f"Output Std = {output_std:.6f}")
            
            if output_std < 1e-6:
                print("⚠️  Warning: Model still outputting constants!")
    
    # Check if Phase 1 worked
    with torch.no_grad():
        all_outputs = []
        for coords, _ in zip(data_patches, data_targets):
            out = model(coords).cpu().numpy().ravel()
            all_outputs.extend(out)
        
        final_std = np.std(all_outputs)
        if final_std < 1e-4:
            print("\n❌ Phase 1 Failed: Model still outputs constants!")
            print("   Trying aggressive fix...")
            
            # Aggressive fix: Add noise to weights
            for param in model.parameters():
                param.data += torch.randn_like(param) * 0.01
        else:
            print(f"\n✓ Phase 1 Success: Output std = {final_std:.6f}")
    
    print("\n" + "="*60)
    print("PHASE 2: Adding Boundary Conditions")
    print("="*60)
    
    # Phase 2: Add boundary conditions
    for epoch in range(min(500, epochs//4), min(1000, epochs//2)):
        batch = exp.sample_batch()
        
        # Boundary loss
        loss_bc = exp.boundary_loss(model, batch).mean()
        
        # Data loss (keep it to prevent collapse)
        data_loss = 0
        for coords, targets in zip(data_patches[::5], data_targets[::5]):  # Use subset
            pred = model(coords).squeeze(-1)
            data_loss += nn.MSELoss()(pred, targets)
        data_loss = data_loss / len(data_patches[::5])
        
        # Combined loss (strong BC weight)
        loss = 1000 * loss_bc + 10 * data_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                test_patch = data_patches[0]
                test_out = model(test_patch).cpu().numpy().ravel()
                output_std = np.std(test_out)
            
            print(f"Epoch {epoch:4d}: BC = {loss_bc.item():.6f}, "
                  f"Data = {data_loss.item():.6f}, Output Std = {output_std:.6f}")
    
    print("\n" + "="*60)
    print("PHASE 3: Full PDE Training")
    print("="*60)
    
    # Phase 3: Add PDE residual
    for epoch in range(min(1000, epochs//2), epochs):
        batch = exp.sample_batch()
        
        # All losses
        loss_bc = exp.boundary_loss(model, batch).mean()
        loss_res = exp.pde_residual_loss(model, batch).mean()
        
        # Keep some data loss to prevent collapse
        data_loss = 0
        for coords, targets in zip(data_patches[::10], data_targets[::10]):
            pred = model(coords).squeeze(-1)
            data_loss += nn.MSELoss()(pred, targets)
        data_loss = data_loss / max(1, len(data_patches[::10]))
        
        # Gradually increase PDE weight
        pde_weight = min(1.0, (epoch - 1000) / 1000) * 0.1
        
        loss = 100 * loss_bc + pde_weight * loss_res + data_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                test_patch = data_patches[0]
                test_out = model(test_patch).cpu().numpy().ravel()
                output_std = np.std(test_out)
                
                # Compute relative L2 error
                rel_l2 = exp.relative_l2_on_grid(model, base_cfg['eval']['grid'])
            
            print(f"Epoch {epoch:4d}: BC = {loss_bc.item():.6f}, "
                  f"Res = {loss_res.item():.6f}, Data = {data_loss.item():.6f}, "
                  f"Std = {output_std:.6f}, Rel L2 = {rel_l2:.4f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Test on grid
        rel_l2 = exp.relative_l2_on_grid(model, base_cfg['eval']['grid'])
        print(f"Final Relative L2 Error: {rel_l2:.6f}")
        
        # Check output distribution
        test_coords = torch.rand(100, 2, device=device) * 2 - 1
        n_test_patches = len(test_coords) // patch_size
        
        all_outputs = []
        for i in range(n_test_patches):
            patch = test_coords[i*patch_size:(i+1)*patch_size]
            out = model(patch).cpu().numpy().ravel()
            all_outputs.extend(out)
        
        all_outputs = np.array(all_outputs)
        print(f"Output Statistics:")
        print(f"  Mean: {all_outputs.mean():.6f}")
        print(f"  Std:  {all_outputs.std():.6f}")
        print(f"  Min:  {all_outputs.min():.6f}")
        print(f"  Max:  {all_outputs.max():.6f}")
        
        if all_outputs.std() < 1e-4:
            print("\n❌ Model still outputs nearly constant values!")
            print("   Suggestions:")
            print("   1. Increase data points")
            print("   2. Use LBFGS optimizer")
            print("   3. Try sin activation function")
            print("   4. Use hard boundary constraints")
        else:
            print("\n✓ Model learned non-constant solution!")
    
    # Save model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"model_patch_{args.model_name}_{timestamp}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="patch_ffn")
    parser.add_argument("--experiment_name", default="helmholtz2d_steady_patch")
    parser.add_argument("--common_config", default="configs/common_config.yaml")
    parser.add_argument("--model_config", default="configs/model/patch_ffn.yaml")
    parser.add_argument("--exp_config", default="configs/experiment/helmholtz2d_steady_patch.yaml")
    
    args = parser.parse_args()
    train_with_data_points(args)