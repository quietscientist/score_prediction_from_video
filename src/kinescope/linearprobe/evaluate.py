"""
LOSO linear probe evaluation on the UDysRS dataset.

Encodes each clip with a (optionally pretrained) PoseViT, then fits a Ridge
regression per leave-one-subject-out fold.  Z-score statistics are computed
from training subjects only within each fold to prevent leakage.

Usage
-----
from kinescope.linearprobe.evaluate import run_linear_probe
results = run_linear_probe(
    data_dir="/path/to/UDysRS_UPDRS_Export",
    pretrained_weights="./pretrain_ckpt/best.pt",
    output_dir="./results/linearprobe",
)
print(results["aggregate"])
"""

import json
import pathlib
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score

from kinescope.linearprobe.udysrs_loader import load_udysrs
from kinescope.pretrain._normalize_clip import normalize_clip


def _encode_clips(arrays: list, encoder, seq_len: int, device) -> np.ndarray:
    """
    Encode a list of variable-length (T, 17, 2) clips into fixed-size embeddings.

    Each clip is:
      1. Normalized via normalize_clip()
      2. Chunked into seq_len-frame windows (stride = seq_len // 2)
      3. Passed through the encoder in a single batch
      4. Chunk embeddings averaged → one (embed_dim,) vector per clip

    Clips shorter than seq_len are zero-padded on the right.
    """
    import torch

    encoder.eval()
    stride = seq_len // 2
    embeddings = []

    with torch.no_grad():
        for arr in arrays:
            normalized = normalize_clip(arr)  # (T, 17, 2)
            T = len(normalized)

            # Chunk into fixed-length windows
            starts = list(range(0, T - seq_len + 1, stride))
            if not starts:
                # Clip shorter than seq_len — pad with zeros
                padded = np.zeros((seq_len, normalized.shape[1], normalized.shape[2]),
                                  dtype=np.float32)
                padded[:T] = normalized
                chunks = [padded]
            else:
                chunks = [normalized[s:s + seq_len] for s in starts]

            # NaN joints (e.g. eyes/ears absent in UDysRS) → 0 before encoder
            batch = torch.tensor(np.nan_to_num(np.stack(chunks), nan=0.0),
                                 dtype=torch.float32).to(device)
            embs = encoder(batch)              # (n_chunks, embed_dim)
            embeddings.append(embs.mean(0).cpu().numpy())

    return np.array(embeddings, dtype=np.float32)


def _zscore_train_stats(scores_raw: np.ndarray, tasks: list, train_mask: np.ndarray):
    """
    Compute per-task z-score statistics from training clips only.

    Returns (N,) z-scored array where each clip is normalized by its task's
    training-set mean and std.  σ → 1.0 if < 1e-6 (constant task).
    """
    task_names = ["drinking", "communication", "la"]
    z = np.zeros_like(scores_raw)
    stats = {}

    for task in task_names:
        task_mask = np.array([t == task for t in tasks])
        train_task = train_mask & task_mask
        vals = scores_raw[train_task]
        mu = float(np.mean(vals)) if len(vals) else 0.0
        sigma = float(np.std(vals)) if len(vals) > 1 else 1.0
        if sigma < 1e-6:
            sigma = 1.0
        stats[task] = (mu, sigma)
        z[task_mask] = (scores_raw[task_mask] - mu) / sigma

    return z, stats


def _plot_scatter(y_true, y_pred, tasks, save_path, title="Combined"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_colors = {"drinking": "#1f77b4", "communication": "#ff7f0e", "la": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for task, color in task_colors.items():
        mask = np.array([t == task for t in tasks])
        ax.scatter(y_true[mask], y_pred[mask], c=color, label=task, alpha=0.6, s=30)

    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlabel("True z-score")
    ax.set_ylabel("Predicted z-score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_per_task(y_true, y_pred, tasks, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    task_names = ["drinking", "communication", "la"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, task_names):
        mask = np.array([t == task for t in tasks])
        yt, yp = y_true[mask], y_pred[mask]
        rho = spearmanr(yt, yp).statistic if len(yt) > 1 else float("nan")
        ax.scatter(yt, yp, alpha=0.5, s=25)
        lims = [min(yt.min(), yp.min()) - 0.2, max(yt.max(), yp.max()) + 0.2]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_title(f"{task}  ρ={rho:.3f}")
        ax.set_xlabel("True z-score")
        ax.set_ylabel("Predicted z-score")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_spearman_by_subject(fold_results, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = [r["subject"] for r in fold_results]
    rhos = [r["spearman"] for r in fold_results]
    order = np.argsort(rhos)

    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.5), 4))
    ax.bar(range(len(subjects)), [rhos[i] for i in order],
           color=["#2ca02c" if rhos[i] >= 0 else "#d62728" for i in order])
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([str(subjects[i]) for i in order], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("LOSO Spearman ρ per subject")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _load_encoder(pretrained_weights: Optional[str], device):
    """
    Load PoseViT encoder from checkpoint.  Reads architecture config from the
    checkpoint itself so you don't need to manually specify embed_dim etc.
    Falls back to default 128/4/4 if config is absent (random-init baseline
    should still pass explicit args).
    """
    import torch
    from kinescope.prediction._vit import PoseViT

    embed_dim, n_layers, n_heads, seq_len = 128, 4, 4, 60  # defaults match pretraining

    if pretrained_weights:
        ckpt = torch.load(pretrained_weights, map_location="cpu")
        cfg = ckpt.get("config", {})
        embed_dim = cfg.get("embed_dim", embed_dim)
        n_layers  = cfg.get("n_layers",  n_layers)
        n_heads   = cfg.get("n_heads",   n_heads)
        seq_len   = cfg.get("seq_len",   seq_len)

    encoder = PoseViT(embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
                      seq_len=seq_len).to(device)

    if pretrained_weights:
        state = ckpt.get("context_encoder", ckpt)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys in checkpoint")
        print(f"Loaded pretrained weights from {pretrained_weights}  "
              f"(embed={embed_dim}, layers={n_layers}, heads={n_heads}, seq={seq_len})")
    else:
        print("No pretrained weights — using random init (baseline)")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder, seq_len


def run_linear_probe(
    data_dir: str,
    pretrained_weights: Optional[str] = None,
    output_dir: str = "results/linearprobe",
    ridge_alpha: float = 1.0,
    device: str = "auto",
    artifacts_dir: str = "artifacts",
) -> dict:
    """
    Run leave-one-subject-out linear probe evaluation on UDysRS.

    Architecture config (embed_dim, n_layers, n_heads, seq_len) is read
    automatically from the checkpoint file.  For a random-init baseline,
    defaults to 128/4/4/60.

    Parameters
    ----------
    data_dir : str
        Path to UDysRS_UPDRS_Export directory (or its parent).
    pretrained_weights : str, optional
        Path to pretrained ViT checkpoint (.pt).  None = random init baseline.
    output_dir : str
        Directory for metrics JSON and diagnostic plots.
    ridge_alpha : float
        Ridge regression regularization strength.
    device : str
        "auto" | "cpu" | "cuda".
    artifacts_dir : str
        Directory for diagnostic PNG plots.

    Returns
    -------
    dict with keys:
      "aggregate"   : dict — mean/std of spearman, pearson, rmse, auroc across folds
      "per_task"    : dict — same metrics computed per task across all folds
      "fold_results": list — per-fold metrics
    """
    import torch

    # ── Device ──────────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # ── Load encoder ────────────────────────────────────────────────────────
    encoder, seq_len = _load_encoder(pretrained_weights, dev)

    # ── Load data ────────────────────────────────────────────────────────────
    data = load_udysrs(data_dir)
    arrays      = data["arrays"]
    scores_raw  = data["scores_raw"]    # (N,) float32
    tasks       = data["task"]          # list[str]
    subject_ids = data["subject_ids"]   # list

    N = len(arrays)
    subject_ids_arr = np.array(subject_ids)
    unique_subjects = sorted(set(subject_ids))
    print(f"\nLOSO over {len(unique_subjects)} subjects, {N} clips total")

    # ── Pre-encode ALL clips once (encoder is frozen) ────────────────────────
    print("Encoding all clips...")
    X_all = _encode_clips(arrays, encoder, seq_len, dev)   # (N, embed_dim)
    print(f"Embeddings: {X_all.shape}")

    # ── LOSO loop ────────────────────────────────────────────────────────────
    fold_results = []
    all_y_true, all_y_pred, all_tasks_test, all_y_raw_test = [], [], [], []

    for subj in unique_subjects:
        train_mask = subject_ids_arr != subj
        test_mask  = subject_ids_arr == subj

        if test_mask.sum() == 0:
            continue

        # Z-score targets using training subjects only (no leakage)
        y_z, _ = _zscore_train_stats(scores_raw, tasks, train_mask)

        X_train, y_train = X_all[train_mask], y_z[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_z[test_mask]
        tasks_test = [tasks[i] for i in range(N) if test_mask[i]]
        raw_test   = scores_raw[test_mask]

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        n_test = len(y_test)
        if n_test < 2:
            rho, r, rmse, auroc = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            rho   = spearmanr(y_test, y_pred).statistic
            r     = pearsonr(y_test, y_pred).statistic
            rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            binary = (raw_test > 0).astype(int)
            auroc  = (float(roc_auc_score(binary, y_pred))
                      if 0 < binary.sum() < n_test else float("nan"))

        fold_results.append({
            "subject": str(subj),
            "n_test":  n_test,
            "spearman": float(rho),
            "pearson":  float(r),
            "rmse":     float(rmse),
            "auroc":    float(auroc),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_tasks_test.extend(tasks_test)
        all_y_raw_test.extend(raw_test.tolist())

        print(f"  subject {subj}: n={n_test:3d}  ρ={rho:+.3f}  AUROC={auroc:.3f}  RMSE={rmse:.3f}")

    # ── Aggregate metrics ────────────────────────────────────────────────────
    def _agg(key):
        vals = [f[key] for f in fold_results if not np.isnan(f[key])]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    aggregate = {
        "auroc":    _agg("auroc"),
        "spearman": _agg("spearman"),
        "pearson":  _agg("pearson"),
        "rmse":     _agg("rmse"),
        "n_folds":  len(fold_results),
    }

    # ── Per-task breakdown (across all folds combined) ───────────────────────
    yt_all  = np.array(all_y_true)
    yp_all  = np.array(all_y_pred)
    raw_all = np.array(all_y_raw_test)
    per_task = {}
    for task in ["drinking", "communication", "la"]:
        mask = np.array([t == task for t in all_tasks_test])
        if mask.sum() < 2:
            per_task[task] = {"spearman": float("nan"), "pearson": float("nan"),
                              "rmse": float("nan"), "auroc": float("nan"), "n": int(mask.sum())}
            continue
        yt_t, yp_t = yt_all[mask], yp_all[mask]
        binary_t   = (raw_all[mask] > 0).astype(int)
        auroc_t    = (float(roc_auc_score(binary_t, yp_t))
                      if 0 < binary_t.sum() < mask.sum() else float("nan"))
        per_task[task] = {
            "auroc":    auroc_t,
            "spearman": float(spearmanr(yt_t, yp_t).statistic),
            "pearson":  float(pearsonr(yt_t, yp_t).statistic),
            "rmse":     float(np.sqrt(mean_squared_error(yt_t, yp_t))),
            "n":        int(mask.sum()),
        }

    print(f"\nAggregate  ρ={aggregate['spearman']['mean']:+.3f}±{aggregate['spearman']['std']:.3f}"
          f"  r={aggregate['pearson']['mean']:+.3f}±{aggregate['pearson']['std']:.3f}"
          f"  RMSE={aggregate['rmse']['mean']:.3f}±{aggregate['rmse']['std']:.3f}"
          f"  AUROC={aggregate['auroc']['mean']:.3f}±{aggregate['auroc']['std']:.3f}")
    for task, m in per_task.items():
        print(f"  {task:15s}  n={m['n']:3d}  ρ={m['spearman']:+.3f}"
              f"  AUROC={m['auroc']:.3f}  RMSE={m['rmse']:.3f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = pathlib.Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "aggregate":    aggregate,
        "per_task":     per_task,
        "fold_results": fold_results,
        "config": {
            "pretrained_weights": str(pretrained_weights),
            "ridge_alpha":        ridge_alpha,
        },
    }
    with open(out_dir / "loso_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_scatter(yt_all, yp_all, all_tasks_test,
                  out_dir / "scatter_combined.png",
                  title=f"LOSO predictions  ρ={aggregate['spearman']['mean']:+.3f}")
    _plot_per_task(yt_all, yp_all, all_tasks_test, out_dir / "scatter_per_task.png")
    _plot_spearman_by_subject(fold_results, out_dir / "spearman_by_subject.png")

    print(f"\nResults saved to {out_dir}/")
    return metrics
