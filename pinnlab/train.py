import os, time, yaml, argparse, math
import torch
import wandb
from tqdm import trange
from pinnlab.registry import get_model, get_experiment
from pinnlab.utils.seed import seed_everything
from pinnlab.utils.early_stopping import EarlyStopping
from pinnlab.utils.plotting import save_plots_1d, save_plots_2d
from pinnlab.utils.wandb_utils import setup_wandb, wandb_log, wandb_finish

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(args):
    base_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    exp_cfg   = load_yaml(args.exp_config)

    # Allow experiment to override in/out dims if needed
    in_features  = exp_cfg.get("in_features", model_cfg.get("in_features"))
    out_features = exp_cfg.get("out_features", model_cfg.get("out_features"))
    model_cfg["in_features"]  = in_features
    model_cfg["out_features"] = out_features

    seed_everything(base_cfg["seed"])

    device = torch.device(base_cfg["device"] if torch.cuda.is_available() else "cpu")
    exp = get_experiment(args.experiment_name)(exp_cfg, device)
    model = get_model(args.model_name)(model_cfg).to(device)

    # Optimizer
    opt_cfg = base_cfg["train"]["optimizer"]
    if opt_cfg["name"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"])
    else:
        raise ValueError("Only Adam is wired in, add more in train.py.")

    # Logging dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_cfg["log"]["out_dir"], f"{args.experiment_name}_{args.model_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # WandB
    run = setup_wandb(base_cfg["log"]["wandb"], args, out_dir, config={
        "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
    })

    # Early stopping
    es_cfg = base_cfg["train"]["early_stopping"]
    early = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"]) if es_cfg["enabled"] else None
    best_state = None
    best_metric = float("inf")

    # Training loop
    w_f = base_cfg["train"]["loss_weights"]["f"]
    w_b = base_cfg["train"]["loss_weights"]["b"]
    w_ic= base_cfg["train"]["loss_weights"]["ic"]

    n_f = base_cfg["train"]["batch"]["n_f"]
    n_b = base_cfg["train"]["batch"]["n_b"]
    n_0 = base_cfg["train"]["batch"]["n_0"]

    epochs = base_cfg["train"]["epochs"]
    pbar = trange(epochs, desc="Training", ncols=100)

    for ep in pbar:
        model.train()
        batch = exp.sample_batch(n_f=n_f, n_b=n_b, n_0=n_0)

        loss_f = exp.pde_residual_loss(model, batch).mean() if batch.get("X_f") is not None else torch.tensor(0., device=device)
        loss_b = exp.boundary_loss(model, batch).mean()        if batch.get("X_b") is not None else torch.tensor(0., device=device)
        loss_0 = exp.initial_loss(model, batch).mean()         if batch.get("X_0") is not None else torch.tensor(0., device=device)

        loss = w_f*loss_f + w_b*loss_b + w_ic*loss_0

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Log
        log_dict = {
            "loss/total": loss.item(),
            "loss/f": loss_f.item(),
            "loss/b": loss_b.item(),
            "loss/ic": loss_0.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": ep
        }
        wandb_log(log_dict)
        pbar.set_postfix({k: f"{v:.3e}" for k,v in log_dict.items() if "loss" in k})

        # Simple validation metric (relative L2 on a fixed grid)
        if ep % 500 == 0 or ep == epochs-1:
            with torch.no_grad():
                rel_l2 = exp.relative_l2_on_grid(model, base_cfg["eval"]["grid"])
                wandb_log({"eval/rel_l2": rel_l2, "epoch": ep})
                if rel_l2 < best_metric:
                    best_metric = rel_l2
                    best_state = {k: v.clone() for k,v in model.state_dict().items()}

            if early and early.step(best_metric):
                print(f"\n[EarlyStopping] Stopping at epoch {ep}. Best rel_l2={best_metric:.3e}")
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation & plots
    model.eval()
    figs = exp.plot_final(model, base_cfg["eval"]["grid"], out_dir)
    for name, path in figs.items():
        wandb_log({f"fig/{name}": wandb.Image(path)})

    wandb_finish()
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--exp_config", required=True)
    args = parser.parse_args()
    main(args)
