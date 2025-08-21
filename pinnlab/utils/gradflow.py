# pinnlab/utils/gradflow.py
import os, json, math, numpy as np, torch
from collections import OrderedDict

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_linear_layers(model):
    """
    Returns:
      layers: OrderedDict[name -> nn.Linear]
      params: list of Parameter in the order [W0, b0, W1, b1, ...]
      layer_param_slices: dict[name -> (start_idx, end_idx)] in params list
    """
    layers = OrderedDict()
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            layers[name] = m

    params = []
    layer_param_slices = {}
    k = 0
    for name, lin in layers.items():
        start = k
        params.append(lin.weight); k += 1
        if lin.bias is not None:
            params.append(lin.bias); k += 1
        end = k
        layer_param_slices[name] = (start, end)
    return layers, params, layer_param_slices


class GradientFlowLogger:
    """
    Collect per-loss gradients per-layer during training, and optionally plot.

    Usage:
      gf = GradientFlowLogger(model, out_dir, enable=True, every=1000,
                              store_vectors=False, wandb_hist=False,
                              plot_cfg={"enabled": True, "kde": True, "stats": True,
                                        "losses": ["bc","res"]})
      ...
      gf.collect({"res": loss_f, "bc": loss_b, "ic": loss_ic})  # before total backward
      ...
      gf.save()
      if gf.plot_enabled:
          gf.save_plots()
    """
    def __init__(self, model, out_dir, enable=True, every=1000,
                 store_vectors=False, max_keep_per_layer=25,
                 wandb_hist=False, device=None, plot_cfg=None):
        self.enable = enable
        self.every = max(1, int(every))
        self.store_vectors = bool(store_vectors)
        self.max_keep = int(max_keep_per_layer)
        self.wandb_hist = bool(wandb_hist)
        self.device = device or next(model.parameters()).device
        self.out_dir = os.path.join(out_dir, "gradflow")
        _ensure_dir(self.out_dir)

        self.layers, self.params, self.layer_slices = list_linear_layers(model)
        self.layer_names = list(self.layers.keys())

        # storage
        self.stats = {}   # {loss_name: {layer_name: {"mean":[], "max":[], "l2":[]}}}
        self.vecs  = {}   # {loss_name: {layer_name: [np.array, ...]}}
        self._counter = 0

        # plotting config (toggle from YAML)
        plot_cfg = plot_cfg or {}
        self.plot_enabled = bool(plot_cfg.get("enabled", False))
        self.plot_kde_enabled = bool(plot_cfg.get("kde", True))
        self.plot_stats_enabled = bool(plot_cfg.get("stats", True))
        self.plot_losses = tuple(plot_cfg.get("losses", ["bc", "res"]))
        self.kde_filename = str(plot_cfg.get("kde_filename", "gradflow_kde.png"))
        self.stats_prefix = str(plot_cfg.get("stats_prefix", "gradflow_stats"))
        self.max_cols = int(plot_cfg.get("max_cols", 4))  # subplots per row

    def should_collect(self, step:int) -> bool:
        return self.enable and (step % self.every == 0)

    @torch.no_grad()
    def _record_stats(self, loss_name, layer_name, g_flat):
        g_abs = g_flat.abs()
        d = self.stats.setdefault(loss_name, {}).setdefault(layer_name, {"mean":[], "max":[], "l2":[]})
        d["mean"].append(float(g_abs.mean()))
        d["max"].append(float(g_abs.max()))
        d["l2"].append(float(torch.linalg.vector_norm(g_flat).item()))

    def collect(self, losses: dict, step: int | None = None):
        """
        losses: dict like {"res": loss_f_scalar, "bc": loss_b_scalar, "ic": loss_ic_scalar}
                You may pass None for a loss you don't use (e.g., 'ic' in steady problems).
        """
        if not self.enable or len(self.params) == 0:
            return

        # decide if we collect this time
        if step is None:
            step = self._counter
            self._counter += 1
        if (step % self.every) != 0:
            return

        for loss_name, loss_val in losses.items():
            # skip missing / non-tensor losses
            if loss_val is None or not torch.is_tensor(loss_val):
                continue
            # reduce to scalar if needed
            if loss_val.dim() > 0:
                loss_val = loss_val.mean()
            # skip if not connected to the graph
            if (not loss_val.requires_grad) or (loss_val.grad_fn is None):
                continue

            grads = torch.autograd.grad(loss_val, self.params,
                                        retain_graph=True, create_graph=False,
                                        allow_unused=True)
            for layer_name, (s, e) in self.layer_slices.items():
                g_parts = [g.reshape(-1) for g in grads[s:e] if g is not None]
                if not g_parts:
                    continue
                g_flat = torch.cat(g_parts, dim=0)
                self._record_stats(loss_name, layer_name, g_flat)

                if self.store_vectors:
                    arr = g_flat.detach().cpu().numpy()
                    L = self.vecs.setdefault(loss_name, {}).setdefault(layer_name, [])
                    if len(L) >= self.max_keep:
                        L.pop(0)
                    L.append(arr)

                if self.wandb_hist:
                    try:
                        import wandb
                        wandb.log({f"gradflow/hist/{loss_name}/{layer_name}":
                                   wandb.Histogram(g_flat.detach().cpu().numpy())})
                    except Exception:
                        pass

    def save(self):
        # save stats as JSON and vectors as NPZ (object arrays to keep ragged)
        stats_path = os.path.join(self.out_dir, "gradflow_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f)

        if self.store_vectors and self.vecs:
            npz_path = os.path.join(self.out_dir, "gradflow_vectors.npz")
            flat = {}
            for ln, per_layer in self.vecs.items():
                for layer, arrs in per_layer.items():
                    flat[f"{ln}|{layer}"] = np.array(arrs, dtype=object)
            np.savez(npz_path, **flat)

        with open(os.path.join(self.out_dir, "layer_names.txt"), "w") as f:
            for n in self.layer_names:
                f.write(n + "\n")

    # --------------- Plotting ---------------
    def save_plots(self):
        """Save KDE (if vectors exist) and/or stats curves into the gradflow dir."""
        if not self.plot_enabled:
            return []
        saved = []
        if self.plot_kde_enabled:
            p = self._plot_kde()
            if p: saved.append(p)
        if self.plot_stats_enabled:
            saved.extend(self._plot_stats_all())
        return saved

    def _plot_kde(self):
        if not self.store_vectors or not self.vecs:
            # nothing to plot as KDE
            return None
        try:
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
                use_sns = True
            except Exception:
                sns = None
                use_sns = False

            layers = self.layer_names
            nL = len(layers)
            ncols = min(self.max_cols, nL) if nL > 0 else 1
            nrows = math.ceil(max(1, nL) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3.0*nrows), squeeze=False)

            for idx, layer in enumerate(layers):
                r, c = divmod(idx, ncols)
                ax = axes[r, c]
                for ln in self.plot_losses:
                    arrs = self.vecs.get(ln, {}).get(layer, [])
                    if not arrs: 
                        continue
                    g = np.asarray(arrs[-1]).ravel()  # last snapshot like the paper
                    if use_sns:
                        sns.kdeplot(g, ax=ax, label=ln.upper(), bw_method="scott", fill=False)
                    else:
                        # fallback: density histogram
                        ax.hist(g, bins=100, density=True, histtype="step", label=ln.upper())
                ax.set_title(layer)
                ax.set_xlim([-3.0, 3.0])
                if idx == 0: ax.legend()

            # hide any empty axes
            for k in range(nL, nrows*ncols):
                r, c = divmod(k, ncols)
                axes[r, c].axis("off")

            plt.tight_layout()
            out = os.path.join(self.out_dir, self.kde_filename)
            fig.savefig(out, dpi=160)
            plt.close(fig)
            return out
        except Exception:
            return None

    def _plot_stats_all(self):
        if not self.stats:
            return []
        try:
            import matplotlib.pyplot as plt
            saved = []
            metrics = ["mean", "max", "l2"]
            for metric in metrics:
                layers = self.layer_names
                nL = len(layers)
                ncols = min(self.max_cols, nL) if nL > 0 else 1
                nrows = math.ceil(max(1, nL) / ncols)
                fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3.0*nrows), squeeze=False)

                for idx, layer in enumerate(layers):
                    r, c = divmod(idx, ncols)
                    ax = axes[r, c]
                    for ln in self.plot_losses:
                        series = self.stats.get(ln, {}).get(layer, {}).get(metric, [])
                        if not series: 
                            continue
                        xs = np.arange(1, len(series)+1)
                        ax.plot(xs, series, label=ln.upper())
                    ax.set_title(f"{layer} | {metric}")
                    ax.set_yscale("symlog")  # like paper figs; robust for wide ranges
                    if idx == 0: ax.legend()

                for k in range(nL, nrows*ncols):
                    r, c = divmod(k, ncols)
                    axes[r, c].axis("off")

                plt.tight_layout()
                out = os.path.join(self.out_dir, f"{self.stats_prefix}_{metric}.png")
                fig.savefig(out, dpi=160)
                plt.close(fig)
                saved.append(out)
            return saved
        except Exception:
            return []
