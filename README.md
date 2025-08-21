# PINN-Lab — A Modular Pipeline for Physics-Informed Neural Networks

> **TL;DR**: Swap models and PDE experiments with a YAML config, train with tidy logs, and auto-plot predictions + errors. Built for fast iteration across classic PDE benchmarks.

---

## ✨ Highlights

- **Pluggable experiments** (1D/2D; steady & time-dependent)  
- **Model zoo**: vanilla MLP, Fourier-feature MLP, ResNet-style PINN (and room for more)  
- **One-line training** via `train.py` with YAML configs  
- **Early stopping & progress bar** out of the box  
- **W&B logging** + **automatic plots** (true/pred/abs-error)  
- **Deterministic seeds** for reproducibility

---

## 📁 Repository Structure

pinnlab/
├─ train.py # CLI entrypoint: loads configs, trains, logs, plots
├─ registry.py # Central registry (models & experiments)
├─ models/ (keep updating)
│ ├─ mlp.py # Baseline MLP
│ ├─ fourier_mlp.py # Fourier features + MLP
│ └─ residual_network.py # Residual (skip-connected) PINN (optional/extend)
├─ experiments/ (keep updating)
│ ├─ base.py
│ ├─ allencahn1d.py
│ ├─ allencahn2d.py
│ ├─ burgers1d.py
│ ├─ convection1d.py
│ ├─ helmholtz2d_steady.py
│ ├─ helmholtz2d.py
│ ├─ navierstokes2d.py
│ ├─ poisson2d.py
│ ├─ reactiondiffusion1d.py
│ └─ reactiondiffusion2d.py
├─ data/
│ ├─ geometries.py # Define simple domain shape (Interval, Rectangle)
│ └─ samplers.py # Sampling data points
└─ utils/
  ├─ early_stopping.py
  ├─ gradflow.py
  ├─ plotting.py
  ├─ seed.py
  └─ wandb_utils.py
configs/
├─ common_config.yaml # global training/log/eval settings
├─ model/.yaml # per-model configs
└─ experiment/.yaml # per-experiment configs
scripts/
└─ model_name/experiment_name.sh # per-model-per-experiment sh files

.
|-- train.py
|-- registry.py
|-- models/
|   |-- mlp.py
|   |-- fourier_mlp.py
|-- experiments/
|   |-- burgers1d.py
|   |-- convection1d.py
|   |-- poisson2d.py
|   |-- helmholtz2d.py
|   |-- navierstokes2d.py
|   |-- reactiondiffusion2d.py
|   |-- allencahn2d.py
|-- utils/
|   |-- early_stopping.py
|   |-- plotting.py
|   |-- seed.py
|   |-- wandb_utils.py
|   |-- geometries.py
|   |-- samplers.py
|-- configs/
|   |-- common_config.yaml


> Tip: The code is deliberately lightweight—add new models or PDEs by dropping a file and registering it in `registry.py`.

---

## 🚀 Quickstart

### 1) Install
pip install -r requirements.txt

### 2) Run experiments
e.g. scripts/mlp/allencahn1d.sh


🧩 Models (with original papers)
- MLP (baseline PINN) — based on the original PINNs framework
Raissi et al., J. Comput. Phys., 2019. arXiv/ADS/Elsevier:
[paper]

- Fourier-feature MLP — random Fourier feature mapping before MLP
Tancik et al., NeurIPS 2020.
[PDF]


Each experiment provides:
- minibatch samplers for interior/boundary/initial points,
- residual/BC/IC losses,
- grid evaluation via relative L2,
- plotting functions that save true/pred/abs-error images.

📊 Logging & Visuals
- W&B: enable in common_config.yaml to log losses, metrics, and final figures automatically.
- Plots: After training, the script saves side-by-side true, pred, and |error| for 1D/2D time slices into the run folder.
- Metric: Relative L2 error on a fixed grid is computed periodically.

🧪 Add a New Experiment (sketch)
1. Create experiments/my_pde.py with a subclass of BaseExperiment.
2. Implement:
    - sample_batch(n_f, n_b, n_0) → dict of tensors for residual/BC/IC
    - pde_residual_loss / boundary_loss / initial_loss → per-point squared residuals
    - relative_l2_on_grid and plot_final
    - Register it in registry.py and add a config under configs/exp.