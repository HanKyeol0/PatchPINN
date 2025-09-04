import os
def setup_wandb(cfg, args, out_dir, config):
    mode = cfg.get("mode","auto")
    enabled = cfg.get("enabled", True)
    if not enabled: mode = "disabled"
    try:
        import wandb
        wandb.init(project=cfg["project"], entity=cfg.get("entity"),
                   config=config, dir=out_dir, mode=mode,
                   name=f"{args.experiment_name}_{args.model_name}")
        return wandb.run
    except Exception as e:
        print("[wandb] disabled:", e)
        class Dummy: pass
        return Dummy()

def wandb_log(d, step=None, commit=True):
    try:
        import wandb
        if step is None:
            wandb.log(d, commit=commit)                # fallback (not recommended)
        else:
            wandb.log(d, step=int(step), commit=commit)
    except Exception:
        pass

def wandb_finish():
    try:
        import wandb; wandb.finish()
    except: pass
