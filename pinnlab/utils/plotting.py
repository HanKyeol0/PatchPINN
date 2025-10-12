import os, csv, numpy as np, torch, matplotlib.pyplot as plt
import imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def plot_weights_over_time(csv_path: str, out_path: str) -> str:
    """
    csv format:
      step,res,bc,ic[,data...]
      0,0.33,0.33,0.34
      1,0.40,0.30,0.30
      ...
    """
    steps, series = [], {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        terms = [h for h in headers if h != "step"]
        for row in reader:
            steps.append(int(row["step"]))
            for t in terms:
                series.setdefault(t, []).append(float(row[t]))
    fig = plt.figure()                              # single-plot only
    for t, ys in series.items():
        plt.plot(steps, ys, label=t)                # no colors specified
    plt.xlabel("Step (epoch)")
    plt.ylabel("Normalized loss weight")
    plt.title("Loss weight evolution")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_triptych_frames_2d(
    x, y,
    U_true_T, U_pred_T, ts,
    out_dir, prefix="frame",
    vmin=None, vmax=None, err_vmax=None,
):
    """
    Save per-time frames with 3 panels: true | pred | abs_error.

    Args:
      x, y: 2D arrays [nx, ny] (meshgrid) or 1D coords (len nx/ny)
      U_true_T, U_pred_T: [T, nx, ny] numpy arrays
      ts: [T] numpy 1D array of time values
      out_dir: directory to write frames
      prefix: filename prefix (e.g., "helmholtz2d")
      vmin/vmax: color scale for true/pred (auto if None)
      err_vmax: color scale max for abs_error (auto if None)

    Returns:
      List of file paths of frames, ordered by time.
    """
    _ensure_dir(out_dir)
    T, nx, ny = U_true_T.shape
    assert U_pred_T.shape == (T, nx, ny)
    xs = x if np.ndim(x) == 1 else x[0, :]
    ys = y if np.ndim(y) == 1 else y[:, 0]
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    if vmin is None or vmax is None:
        global_min = np.min([U_true_T.min(), U_pred_T.min()])
        global_max = np.max([U_true_T.max(), U_pred_T.max()])
        vmin = global_min if vmin is None else vmin
        vmax = global_max if vmax is None else vmax
    if err_vmax is None:
        err_vmax = np.abs(U_true_T - U_pred_T).max()

    frame_paths = []
    for i in range(T):
        ut = U_true_T[i]
        up = U_pred_T[i]
        err = np.abs(ut - up)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        ims = []
        ims.append(axes[0].imshow(ut, origin="lower", extent=extent, vmin=vmin, vmax=vmax))
        axes[0].set_title("True")
        ims.append(axes[1].imshow(up, origin="lower", extent=extent, vmin=vmin, vmax=vmax))
        axes[1].set_title("Pred")
        ims.append(axes[2].imshow(err, origin="lower", extent=extent, vmin=0.0, vmax=err_vmax))
        axes[2].set_title("|True − Pred|")

        for ax in axes:
            ax.set_xlabel("x"); ax.set_ylabel("y")

        # colorbars (kept compact)
        for ax, im in zip(axes, ims):
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)

        fig.suptitle(f"t = {ts[i]:.4f}")
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(path)

    return frame_paths

def write_video_from_frames(frame_paths, out_path, fps=10):
    """
    Assemble frames into a video. Prefers imageio; falls back to Matplotlib.
    If MP4 is requested but no encoder is available, transparently saves a GIF and returns its path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()

    # 1) Try imageio (mp4/gif if installed)
    try:
        import imageio.v3 as iio
        imgs = [iio.imread(p) for p in frame_paths]
        iio.imwrite(out_path, imgs, fps=fps)  # mp4 needs imageio-ffmpeg; gif uses pillow plugin
        return out_path
    except Exception:
        try:
            import imageio
            with imageio.get_writer(out_path, fps=fps) as w:
                for p in frame_paths:
                    w.append_data(imageio.v2.imread(p))
            return out_path
        except Exception:
            pass  # will try Matplotlib

    # 2) Matplotlib fallback
    fig = plt.figure()
    ims = [[plt.imshow(plt.imread(p), animated=True)] for p in frame_paths]
    ani = animation.ArtistAnimation(fig, ims, interval=int(1000 / fps), blit=True)

    if ext == ".mp4":
        if _has_ffmpeg():
            ani.save(out_path, writer="ffmpeg", fps=fps)
            plt.close(fig)
            return out_path
        else:
            alt = os.path.splitext(out_path)[0] + ".gif"
            ani.save(alt, writer="pillow", fps=fps)   # always supported by Pillow
            plt.close(fig)
            print(f"[video] ffmpeg/imageio not available; saved GIF instead: {alt}")
            return alt
    else:
        # .gif / .apng etc.
        ani.save(out_path, writer="pillow", fps=fps)
        plt.close(fig)
        return out_path

def _has_ffmpeg():
    try:
        return animation.writers.is_available("ffmpeg")
    except Exception:
        return False

def save_video_2d(
    x, y, U_true_T, U_pred_T, ts,
    out_path, tmp_dir=None, fps=10,
    vmin=None, vmax=None, err_vmax=None, prefix="frame",
):
    """
    Convenience wrapper: renders triptych frames then writes video.

    Args:
      x, y: meshgrid arrays [nx, ny] or 1D coords (nx) & (ny)
      U_true_T, U_pred_T: [T, nx, ny] numpy arrays
      ts: [T] 1D numpy array of time values
      out_path: path to .mp4 or .gif
      tmp_dir: optional directory to place frames (default: alongside out_path/_frames)
    """
    if tmp_dir is None:
        base = os.path.splitext(out_path)[0]
        tmp_dir = base + "_frames"
    frames = save_triptych_frames_2d(
        x, y, U_true_T, U_pred_T, ts,
        out_dir=tmp_dir, prefix=prefix,
        vmin=vmin, vmax=vmax, err_vmax=err_vmax,
    )
    return write_video_from_frames(frames, out_path)