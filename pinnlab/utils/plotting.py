import os, numpy as np, torch, matplotlib.pyplot as plt

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_plots_1d(x, t, u_true, u_pred, out_dir, prefix):
    _ensure_dir(out_dir)
    # pick a few time slices
    ts = np.linspace(t.min(), t.max(), 4)
    paths = {}
    for i, ti in enumerate(ts):
        idx = np.argmin(np.abs(t - ti))
        fig = plt.figure()
        plt.plot(x, u_true[:, idx], label="true")
        plt.plot(x, u_pred[:, idx], linestyle="--", label="pred")
        plt.title(f"{prefix}  t={t[idx]:.3f}")
        plt.xlabel("x"); plt.ylabel("u"); plt.legend()
        path = os.path.join(out_dir, f"{prefix}_t{i}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths[f"{prefix}_t{i}"] = path

    # absolute error heatmap
    err = np.abs(u_true - u_pred)
    fig = plt.figure()
    plt.imshow(err, origin="lower", aspect="auto",
               extent=[t.min(), t.max(), x.min(), x.max()])
    plt.colorbar(label="|u_true - u_pred|")
    plt.xlabel("t"); plt.ylabel("x"); plt.title(f"{prefix} abs error")
    path = os.path.join(out_dir, f"{prefix}_abs_error.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    paths[f"{prefix}_abs_error"] = path
    return paths

def save_plots_2d(x, y, u_true, u_pred, out_dir, prefix):
    _ensure_dir(out_dir)
    err = np.abs(u_true - u_pred)
    paths = {}
    for name, arr in [("true", u_true), ("pred", u_pred), ("abs_error", err)]:
        fig = plt.figure()
        plt.imshow(arr, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(label=name)
        plt.xlabel("x"); plt.ylabel("y"); plt.title(f"{prefix} {name}")
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths[f"{prefix}_{name}"] = path
    return paths
