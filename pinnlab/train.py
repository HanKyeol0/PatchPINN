import os, time, yaml, argparse, sys
import torch
import wandb
import csv
from tqdm import trange
from pinnlab.registry import get_model, get_experiment
from pinnlab.utils.seed import seed_everything
from pinnlab.utils.early_stopping import EarlyStopping
from pinnlab.utils.plotting import save_plots_1d, save_plots_2d
from pinnlab.utils.wandb_utils import setup_wandb, wandb_log, wandb_finish
from pinnlab.utils.gradflow import GradientFlowLogger
from pinnlab.utils.loss_balancer import BalancerConfig, make_loss_balancer
from pinnlab.utils.plotting import plot_weights_over_time

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)

def main(args):
    base_cfg = load_yaml(args.common_config)
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

    # Logging dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_cfg["log"]["out_dir"], f"{args.experiment_name}_{args.model_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    _save_yaml(os.path.join(out_dir, "config.yaml"), {
        "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
    })
    
    # Loss balancer
    use_loss_balancer = base_cfg["train"]["loss_balancer"].get("use_loss_balancer", False)
    if use_loss_balancer:
        lb_cfg_dict = base_cfg["train"].get("loss_balancer", {})  # {'kind': 'dwa', 'terms': ['res','bc','ic'], ...}
        lb_cfg = BalancerConfig(**lb_cfg_dict)
        if not lb_cfg.terms:
            lb_cfg.terms = ["res", "bc", "ic"]   # tailor per experiment

        balancer = make_loss_balancer(lb_cfg)

    weights_csv = os.path.join(out_dir, "loss_weights.csv")
    weights_terms = None  # will infer on first log

    # Optimizer
    params = list(model.parameters())
    if use_loss_balancer:
       params += list(balancer.extra_params())  # no-op for other schemes

    opt_cfg = base_cfg["train"]["optimizer"]
    if opt_cfg["name"].lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.0))
    else:
        raise ValueError("Only Adam is wired in, add more in train.py.")

    # create file with header late (when we know the terms)
    def _ensure_weights_header(terms):
        nonlocal weights_terms
        if weights_terms is None:
            weights_terms = list(terms)
            with open(weights_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + weights_terms)

    global_step = 0

    # WandB
    if base_cfg["log"]["wandb"]["enabled"]:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(project = base_cfg["log"]["wandb"]["project"],
                   name = f"{args.experiment_name}_{args.model_name}_{ts}")
        run = setup_wandb(base_cfg["log"]["wandb"], args, out_dir, config={
            "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
        })

    # gradient flow logger
    gf = GradientFlowLogger(
        model, out_dir,
        enable=base_cfg.get("gradflow", {}).get("enabled", True),
        every=int(base_cfg.get("gradflow", {}).get("every", 1000)),
        store_vectors=bool(base_cfg.get("gradflow", {}).get("store_vectors", False)),
        max_keep_per_layer=int(base_cfg.get("gradflow", {}).get("max_keep", 20)),
        wandb_hist=bool(base_cfg.get("gradflow", {}).get("wandb_hist", False)),
        plot_cfg=base_cfg.get("gradflow", {}).get("plot", {})
    )

    epochs = base_cfg["train"]["epochs"]
    eval_every = int(base_cfg.get("eval").get("every", 100))

    # Early stopping
    es_cfg = base_cfg["train"]["early_stopping"]
    early = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"], eval_every=eval_every) if es_cfg["enabled"] else None
    best_state = None
    best_metric = float("inf")

    # Training loop
    w_res = base_cfg["train"]["loss_weights"]["res"]
    w_bc = base_cfg["train"]["loss_weights"]["bc"]
    w_ic = base_cfg["train"]["loss_weights"]["ic"]

    n_f = exp_cfg.get("batch", {}).get("n_f", base_cfg["train"]["batch"]["n_f"])
    n_b = exp_cfg.get("batch", {}).get("n_b", base_cfg["train"]["batch"]["n_b"])
    n_0 = exp_cfg.get("batch", {}).get("n_0", base_cfg["train"]["batch"]["n_0"])

    use_tty = sys.stdout.isatty()
    pbar = trange(
        epochs,
        desc="Training",
        ncols=120,
        dynamic_ncols=True,
        leave=False,          # don't leave old bars behind
        disable=not use_tty,  # if output is piped, avoid multiline spam
    )
    gf_stop = base_cfg["gradflow"]["stop_at"]

    patches = exp.sample_patches() # {"X_f": x_f, "X_b": x_b, "u_b": u_b}

    for ep in pbar:
        model.train()
        # batch = exp.sample_batch(n_f=n_f, n_b=n_b, n_0=n_0)

        loss_res = exp.pde_residual_loss(model, patches).mean() if patches.get("X_f") is not None else torch.tensor(0., device=device)
        loss_bc = exp.boundary_loss(model, patches).mean()     if patches.get("X_b") is not None else torch.tensor(0., device=device)
        loss_ic = exp.initial_loss(model, patches).mean()      if patches.get("X_0") is not None else torch.tensor(0., device=device)

        loss_res_s = loss_res.mean() if torch.is_tensor(loss_res) and loss_res.dim() > 0 else loss_res # scalar
        loss_bc_s = loss_bc.mean() if torch.is_tensor(loss_bc) and loss_bc.dim() > 0 else loss_bc
        loss_ic_s = loss_ic.mean() if torch.is_tensor(loss_ic) and loss_ic.dim() > 0 else loss_ic

        if ep <= gf_stop:
            gf.collect({"res": loss_res_s, "bc": loss_bc_s, "ic": loss_ic_s}, step=global_step)

        losses = {
            "res": loss_res_s,     # PDE residual term
            **({"bc": loss_bc} if "loss_bc" in locals() else {}),
            **({"ic": loss_ic} if "loss_ic" in locals() else {}),
            # **({"data": loss_data} if "loss_data" in locals() else {}),
        }

        if not use_loss_balancer:
            total_loss = w_res*loss_res + w_bc*loss_bc + w_ic*loss_ic
            s = (w_res + w_bc + w_ic) or 1.0
            w_now = {"res": w_res/s, "bc": w_bc/s, "ic": w_ic/s}
        else:
            total_loss, w_dict, aux = balancer(losses, step=global_step, model=model)
            w_now = {k.split("/", 1)[1]: float(v) for k, v in w_dict.items()}

        # write one row per epoch/step
        _ensure_weights_header(w_now.keys())
        with open(weights_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([global_step] + [w_now[t] for t in weights_terms])

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        global_step += 1

        # Log
        log_dict = {
            "loss/total": total_loss.item(),
            "loss/res": loss_res.item(),
            "loss/bc": loss_bc.item(),
            "loss/ic": loss_ic.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": ep
        }
        wandb_log(log_dict, step=global_step, commit=False)
        pbar.set_postfix({k: f"{v:.3e}" for k,v in log_dict.items() if "loss" in k})
        log_payload = {"loss/total": float(total_loss.detach().cpu())}
        log_payload.update(w_dict)
        log_payload.update(aux)        # e.g., sigma values for uncertainty scheme
        for k, v in losses.items():
            log_payload[f"loss/{k}"] = float(v.detach().cpu())
        wandb_log(log_payload, step=global_step, commit=True)

        # Simple validation metric (relative L2 on a fixed grid)
        best_path = os.path.join(out_dir, "best.pt")
        if ep % eval_every == 0 or ep == epochs-1:
            with torch.no_grad():
                rel_l2 = exp.relative_l2_on_grid(model, base_cfg["eval"]["grid"])
            wandb_log({"eval/rel_l2": rel_l2, "epoch": ep}, step=global_step)

            if rel_l2 < (best_metric - es_cfg.get("min_delta", 0.0)):
                best_metric = rel_l2
                best_state = {k: v.clone() for k,v in model.state_dict().items()}
                torch.save({k:v.detach().cpu() for k,v in best_state.items()}, best_path)

            if early and early.step(rel_l2):
                print(f"\n[EarlyStopping] Stopping at epoch {ep}. Best rel_l2={best_metric:.3e}")
                break

    weights_png = os.path.join(out_dir, "loss_weights.png")
    plot_weights_over_time(weights_csv, weights_png)
    print(f"[weights] saved: {weights_csv}")
    print(f"[weights] plot : {weights_png}")

    gf.save()
    if gf.plot_enabled:
        saved = gf.save_plots()
        for p in saved:
            print(f"[gradflow] saved plot: {p}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation & plots
    model.eval()
    figs = exp.plot_final(model, base_cfg["eval"]["grid"], out_dir)
    for name, path in figs.items():
        wandb_log({f"fig/{name}": wandb.Image(path)}, step=global_step)

    wandb_finish()
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--common_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--exp_config", required=True)
    args = parser.parse_args()
    main(args)