"""
LOSO linear probe evaluation on the UDysRS dataset.

Runs four evaluations in sequence:

  1. Normalization audit — checks whether shoulder sway (whole-body postural
     drift, preserved by clip-mean centering in normalize_clip) correlates with
     score, and whether wrist/ankle speed in normalized space also correlates.
     Verifies that the normalization strategy retains dyskinesia-relevant signals.

  2. Kinematic feature baseline — LOSO Ridge on 17 handcrafted features
     (limb speed/variance, bilateral asymmetry, upper-lower coordination,
     smoothness/jerk, range of motion) computed from normalized arrays.
     Sets the floor: if the encoder doesn't beat this, pretraining isn't helping.

  3. Combined probe — LOSO Ridge on frozen encoder embeddings, all tasks
     combined with per-task z-scored targets.

  4. Per-task probes — same probe repeated separately for drinking,
     communication, and leg agility.  Removes cross-task confounding.

Usage
-----
from kinescope.linearprobe.evaluate import run_linear_probe
results = run_linear_probe(
    data_dir="/path/to/UDysRS_UPDRS_Export",
    pretrained_weights="./pretrain_ckpt/best.pt",
    output_dir="./results/linearprobe",
)
print(results["combined"]["aggregate"])
"""

import json
import pathlib
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from kinescope.linearprobe.udysrs_loader import load_udysrs
from kinescope.pretrain._normalize_clip import normalize_clip


# ── Correlation helper ────────────────────────────────────────────────────────

def _safe_corr(a: np.ndarray, b: np.ndarray):
    """Pearson r and Spearman ρ between 1D arrays, handling NaN and constants."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    a, b = a[mask], b[mask]
    r = (float(pearsonr(a, b).statistic)
         if np.std(a) > 1e-8 and np.std(b) > 1e-8 else float("nan"))
    rho = float(spearmanr(a, b).statistic)
    return r, rho


# ── Encoder loading ───────────────────────────────────────────────────────────

def _load_encoder(pretrained_weights: Optional[str], device):
    """
    Load PoseViT from checkpoint.  Reads architecture from checkpoint config so
    you don't need to specify embed_dim etc. manually.  Falls back to 128/4/4/60
    (default pretraining config) for random-init baseline.
    """
    import torch
    from kinescope.prediction._vit import PoseViT

    embed_dim, n_layers, n_heads, seq_len = 128, 4, 4, 60

    ckpt = None
    if pretrained_weights:
        ckpt = torch.load(pretrained_weights, map_location="cpu")
        cfg = ckpt.get("config", {})
        embed_dim = cfg.get("embed_dim", embed_dim)
        n_layers  = cfg.get("n_layers",  n_layers)
        n_heads   = cfg.get("n_heads",   n_heads)
        seq_len   = cfg.get("seq_len",   seq_len)

    encoder = PoseViT(embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
                      seq_len=seq_len).to(device)

    if ckpt is not None:
        state = ckpt.get("context_encoder", ckpt)
        missing, _ = encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys in checkpoint")
        print(f"Loaded pretrained weights from {pretrained_weights}  "
              f"(embed={embed_dim}, layers={n_layers}, heads={n_heads}, seq={seq_len})")
    else:
        print("No pretrained weights — using random init (baseline, 128/4/4/60)")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder, seq_len


# ── Clip encoding ─────────────────────────────────────────────────────────────

def _encode_clips(arrays: list, encoder, seq_len: int, device) -> np.ndarray:
    """
    Encode a list of variable-length (T, 17, 2) clips into (N, embed_dim) embeddings.

    Each clip is normalized, chunked into seq_len windows (stride = seq_len//2),
    encoded, then chunk embeddings are averaged.  Clips shorter than seq_len
    are zero-padded.
    """
    import torch

    encoder.eval()
    stride = seq_len // 2
    embeddings = []

    with torch.no_grad():
        for arr in arrays:
            normalized = normalize_clip(np.nan_to_num(arr, nan=0.0))
            T = len(normalized)

            starts = list(range(0, T - seq_len + 1, stride))
            if not starts:
                padded = np.zeros((seq_len, 17, 2), dtype=np.float32)
                padded[:T] = normalized
                chunks = [padded]
            else:
                chunks = [normalized[s:s + seq_len] for s in starts]

            batch = torch.tensor(np.stack(chunks), dtype=torch.float32).to(device)
            embs = encoder(batch)               # (n_chunks, embed_dim)
            embeddings.append(embs.mean(0).cpu().numpy())

    return np.array(embeddings, dtype=np.float32)


# ── Kinematic feature baseline ────────────────────────────────────────────────

def _compute_kinematic_features(arrays: list) -> np.ndarray:
    """
    Compute 17 handcrafted kinematic features from raw (T, 17, 2) arrays.
    normalize_clip is applied internally.  Returns (N, 17) float32.

    Features (all in trunk-length-normalized coordinates):
      [0-3]  mean speed: left_wrist, right_wrist, left_ankle, right_ankle
      [4-7]  speed variance: same joints
      [8]    wrist bilateral asymmetry  |mean_lw - mean_rw|
      [9]    ankle bilateral asymmetry  |mean_la - mean_ra|
      [10]   wrist bilateral correlation (symmetry of timing)
      [11]   ankle bilateral correlation
      [12]   upper-lower coordination (wrist speed vs. ankle speed)
      [13]   wrist smoothness: mean |Δspeed| (jerk proxy)
      [14]   ankle smoothness
      [15]   wrist range of motion (std of x position, both wrists summed)
      [16]   ankle range of motion (std of y position, both ankles summed)
    """
    def _speeds(arr_norm, idx):
        v = np.linalg.norm(np.diff(arr_norm[:, idx, :], axis=0), axis=1)
        return v

    def _pearson_or_zero(a, b):
        if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(pearsonr(a, b).statistic)

    rows = []
    for arr in arrays:
        arr_norm = normalize_clip(np.nan_to_num(arr, nan=0.0))

        lw = _speeds(arr_norm, 9)
        rw = _speeds(arr_norm, 10)
        la = _speeds(arr_norm, 15)
        ra = _speeds(arr_norm, 16)

        feat = [
            # mean speed
            float(np.nanmean(lw)), float(np.nanmean(rw)),
            float(np.nanmean(la)), float(np.nanmean(ra)),
            # speed variance
            float(np.nanvar(lw)), float(np.nanvar(rw)),
            float(np.nanvar(la)), float(np.nanvar(ra)),
            # bilateral asymmetry
            abs(float(np.nanmean(lw)) - float(np.nanmean(rw))),
            abs(float(np.nanmean(la)) - float(np.nanmean(ra))),
            # bilateral symmetry correlation
            _pearson_or_zero(lw, rw),
            _pearson_or_zero(la, ra),
            # upper-lower coordination
            _pearson_or_zero(0.5 * (lw + rw), 0.5 * (la + ra)),
            # smoothness (jerk proxy: mean |Δspeed|)
            float(np.nanmean(np.abs(np.diff(lw)))) + float(np.nanmean(np.abs(np.diff(rw)))),
            float(np.nanmean(np.abs(np.diff(la)))) + float(np.nanmean(np.abs(np.diff(ra)))),
            # range of motion
            float(np.nanstd(arr_norm[:, 9, 0]) + np.nanstd(arr_norm[:, 10, 0])),  # wrist-x
            float(np.nanstd(arr_norm[:, 15, 1]) + np.nanstd(arr_norm[:, 16, 1])), # ankle-y
        ]
        rows.append(np.array(feat, dtype=np.float32))

    return np.array(rows, dtype=np.float32)  # (N, 17)


# ── Normalization audit ───────────────────────────────────────────────────────

def _shoulder_sway(arr: np.ndarray) -> float:
    """
    Trunk-length-normalized shoulder midpoint displacement (std over time).

    Measures whole-body postural sway/drift.  normalize_clip uses clip-mean
    centering so this signal IS preserved in the normalized representation.
    Correlation with score validates that the normalization retains it.
    """
    l_sh, r_sh = arr[:, 5, :], arr[:, 6, :]
    l_hp, r_hp = arr[:, 11, :], arr[:, 12, :]
    shoulder_mid = (l_sh + r_sh) / 2.0         # (T, 2)
    hip_mid      = (l_hp + r_hp) / 2.0
    trunk_len    = np.linalg.norm(shoulder_mid - hip_mid, axis=1)  # (T,)
    median_trunk = float(np.nanmedian(trunk_len))
    if median_trunk < 1e-6 or not np.isfinite(median_trunk):
        return float("nan")
    sway_xy = np.nanstd(shoulder_mid, axis=0)   # (2,)
    return float(np.sqrt((sway_xy ** 2).sum())) / median_trunk


def normalization_audit(arrays: list, scores_raw: np.ndarray, tasks: list) -> dict:
    """
    Verify that normalize_clip retains dyskinesia-relevant signals.

    Per-task, reports:
      - Shoulder sway correlation with score: normalize_clip uses clip-mean
        centering so sway IS preserved.  High ρ validates this choice.
      - Wrist / ankle speed correlation with score in normalized space:
        checks that relative-joint movement also discriminates score.
      - High-vs-low score speed ratio: mean speed in top-25% vs. bottom-25%
        score clips.  Ratio > 1.2 suggests good discriminability.

    Printed to stdout and returned as a dict (included in metrics.json).
    """
    print("\n=== Normalization Audit ===")
    print("    (shoulder sway is preserved by clip-mean centering in normalize_clip)")

    # Pre-compute per-clip signals
    sways, wrist_sp, ankle_sp = [], [], []
    for arr in arrays:
        sways.append(_shoulder_sway(arr))
        arr_norm = normalize_clip(np.nan_to_num(arr, nan=0.0))
        lw = np.linalg.norm(np.diff(arr_norm[:, 9,  :], axis=0), axis=1)
        rw = np.linalg.norm(np.diff(arr_norm[:, 10, :], axis=0), axis=1)
        la = np.linalg.norm(np.diff(arr_norm[:, 15, :], axis=0), axis=1)
        ra = np.linalg.norm(np.diff(arr_norm[:, 16, :], axis=0), axis=1)
        wrist_sp.append(float(np.nanmean(0.5 * (lw + rw))))
        ankle_sp.append(float(np.nanmean(0.5 * (la + ra))))

    sways     = np.array(sways,     dtype=np.float32)
    wrist_sp  = np.array(wrist_sp,  dtype=np.float32)
    ankle_sp  = np.array(ankle_sp,  dtype=np.float32)

    audit = {}
    for task in ["drinking", "communication", "la"]:
        mask = np.array([t == task for t in tasks])
        if mask.sum() < 5:
            continue

        s = scores_raw[mask]
        r_sway,  rho_sway  = _safe_corr(sways[mask],    s)
        r_wrist, rho_wrist = _safe_corr(wrist_sp[mask], s)
        r_ankle, rho_ankle = _safe_corr(ankle_sp[mask], s)

        # Speed ratio: top-25% score clips vs. bottom-25%
        q25, q75 = np.percentile(s[np.isfinite(s)], [25, 75])
        lo = mask & (scores_raw <= q25)
        hi = mask & (scores_raw >= q75)
        def _ratio(arr_feat):
            lo_mean = float(np.nanmean(arr_feat[lo])) if lo.sum() > 1 else float("nan")
            hi_mean = float(np.nanmean(arr_feat[hi])) if hi.sum() > 1 else float("nan")
            return hi_mean / lo_mean if lo_mean > 1e-8 else float("nan")

        wrist_ratio = _ratio(wrist_sp)
        ankle_ratio = _ratio(ankle_sp)
        sway_ratio  = _ratio(sways)

        sway_flag = " ← low sway correlation despite clip-mean centering" if abs(rho_sway) < 0.10 else ""

        print(f"\n  {task} (n={mask.sum()})")
        print(f"    Shoulder sway (preserved):        ρ={rho_sway:+.3f}  r={r_sway:+.3f}"
              f"  hi/lo={sway_ratio:.2f}x{sway_flag}")
        print(f"    Wrist speed   (post-norm):        ρ={rho_wrist:+.3f}  r={r_wrist:+.3f}"
              f"  hi/lo={wrist_ratio:.2f}x")
        print(f"    Ankle speed   (post-norm):        ρ={rho_ankle:+.3f}  r={r_ankle:+.3f}"
              f"  hi/lo={ankle_ratio:.2f}x")

        audit[task] = {
            "n": int(mask.sum()),
            "shoulder_sway_rho":      rho_sway,
            "shoulder_sway_r":        r_sway,
            "shoulder_sway_hi_lo":    sway_ratio,
            "wrist_speed_rho":        rho_wrist,
            "wrist_speed_r":          r_wrist,
            "wrist_speed_hi_lo":      wrist_ratio,
            "ankle_speed_rho":        rho_ankle,
            "ankle_speed_r":          r_ankle,
            "ankle_speed_hi_lo":      ankle_ratio,
        }

    return audit


# ── Z-score helper ────────────────────────────────────────────────────────────

def _zscore_train_stats(scores_raw: np.ndarray, tasks: list, train_mask: np.ndarray):
    """
    Per-task z-score using only training-fold statistics (no leakage).
    Returns (N,) z-scored array and per-task (mean, std) dict.
    """
    z = np.zeros_like(scores_raw)
    stats = {}
    for task in ["drinking", "communication", "la"]:
        tmask = np.array([t == task for t in tasks])
        vals  = scores_raw[train_mask & tmask]
        mu    = float(np.mean(vals))    if len(vals) else 0.0
        sigma = float(np.std(vals))     if len(vals) > 1 else 1.0
        sigma = sigma if sigma > 1e-6 else 1.0
        stats[task] = (mu, sigma)
        z[tmask] = (scores_raw[tmask] - mu) / sigma
    return z, stats


# ── LOSO loop ─────────────────────────────────────────────────────────────────

def _run_loso(
    X_all: np.ndarray,
    scores_raw: np.ndarray,
    tasks: list,
    subject_ids,
    ridge_alpha: float,
    label: str = "",
) -> dict:
    """
    Run leave-one-subject-out Ridge regression and return a results dict.

    Prints per-subject and aggregate summary.  Returns a dict with keys:
    fold_results, all_y_true, all_y_pred, all_tasks_test, all_y_raw_test,
    aggregate, per_task.
    """
    subject_ids_arr  = np.array(subject_ids)
    unique_subjects  = sorted(set(subject_ids))
    N = len(scores_raw)

    fold_results = []
    all_y_true, all_y_pred, all_tasks_test, all_y_raw_test = [], [], [], []

    for subj in unique_subjects:
        train_mask = subject_ids_arr != subj
        test_mask  = subject_ids_arr == subj
        if test_mask.sum() == 0:
            continue

        y_z, _ = _zscore_train_stats(scores_raw, tasks, train_mask)
        X_train, y_train = X_all[train_mask], y_z[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_z[test_mask]
        tasks_test = [tasks[i] for i in range(N) if test_mask[i]]
        raw_test   = scores_raw[test_mask]

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        n = len(y_test)
        if n < 2:
            rho, r, rmse, auroc = (float("nan"),) * 4
        else:
            rho   = float(spearmanr(y_test, y_pred).statistic)
            r     = float(pearsonr(y_test,  y_pred).statistic)
            rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            binary = (raw_test > 0).astype(int)
            auroc  = (float(roc_auc_score(binary, y_pred))
                      if 0 < binary.sum() < n else float("nan"))

        fold_results.append(dict(subject=str(subj), n_test=n,
                                 spearman=rho, pearson=r, rmse=rmse, auroc=auroc))
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_tasks_test.extend(tasks_test)
        all_y_raw_test.extend(raw_test.tolist())

    # Aggregate
    def _agg(key):
        vals = [f[key] for f in fold_results if np.isfinite(f[key])]
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    aggregate = {k: _agg(k) for k in ("auroc", "spearman", "pearson", "rmse")}
    aggregate["n_folds"] = len(fold_results)

    # Per-task
    yt  = np.array(all_y_true)
    yp  = np.array(all_y_pred)
    raw = np.array(all_y_raw_test)
    per_task = {}
    for task in ["drinking", "communication", "la"]:
        m = np.array([t == task for t in all_tasks_test])
        if m.sum() < 2:
            per_task[task] = dict(spearman=float("nan"), pearson=float("nan"),
                                  rmse=float("nan"), auroc=float("nan"), n=int(m.sum()))
            continue
        yt_t, yp_t  = yt[m], yp[m]
        bin_t = (raw[m] > 0).astype(int)
        auroc_t = (float(roc_auc_score(bin_t, yp_t))
                   if 0 < bin_t.sum() < m.sum() else float("nan"))
        per_task[task] = dict(
            auroc    = auroc_t,
            spearman = float(spearmanr(yt_t, yp_t).statistic),
            pearson  = float(pearsonr(yt_t,  yp_t).statistic),
            rmse     = float(np.sqrt(mean_squared_error(yt_t, yp_t))),
            n        = int(m.sum()),
        )

    hdr = f"[{label}] " if label else ""
    agg = aggregate
    print(f"\n{hdr}Aggregate ({len(fold_results)} subjects)")
    print(f"  ρ={agg['spearman']['mean']:+.3f}±{agg['spearman']['std']:.3f}"
          f"  AUROC={agg['auroc']['mean']:.3f}±{agg['auroc']['std']:.3f}"
          f"  RMSE={agg['rmse']['mean']:.3f}±{agg['rmse']['std']:.3f}")
    for task, m in per_task.items():
        print(f"  {task:15s}  n={m['n']:3d}  ρ={m['spearman']:+.3f}"
              f"  AUROC={m['auroc']:.3f}  RMSE={m['rmse']:.3f}")

    return dict(
        fold_results   = fold_results,
        all_y_true     = all_y_true,
        all_y_pred     = all_y_pred,
        all_tasks_test = all_tasks_test,
        all_y_raw_test = all_y_raw_test,
        aggregate      = aggregate,
        per_task       = per_task,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_scatter(y_true, y_pred, tasks, save_path, title=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"drinking": "#1f77b4", "communication": "#ff7f0e", "la": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for task, c in colors.items():
        m = np.array([t == task for t in tasks])
        ax.scatter(y_true[m], y_pred[m], c=c, label=task, alpha=0.6, s=30)
    lim = [min(y_true.min(), y_pred.min()) - 0.2, max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("True z-score"); ax.set_ylabel("Predicted z-score")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def _plot_per_task(y_true, y_pred, tasks, save_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, ["drinking", "communication", "la"]):
        m = np.array([t == task for t in tasks])
        yt, yp = y_true[m], y_pred[m]
        rho = float(spearmanr(yt, yp).statistic) if len(yt) > 1 else float("nan")
        ax.scatter(yt, yp, alpha=0.5, s=25)
        lim = [min(yt.min(), yp.min()) - 0.2, max(yt.max(), yp.max()) + 0.2]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(f"{task}  ρ={rho:.3f}")
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def _plot_spearman_by_subject(fold_results, save_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = [r["subject"] for r in fold_results]
    rhos     = [r["spearman"] for r in fold_results]
    order    = np.argsort(rhos)
    fig, ax  = plt.subplots(figsize=(max(6, len(subjects) * 0.5), 4))
    ax.bar(range(len(subjects)), [rhos[i] for i in order],
           color=["#2ca02c" if rhos[i] >= 0 else "#d62728" for i in order])
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([str(subjects[i]) for i in order], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Spearman ρ"); ax.set_title("LOSO Spearman ρ per subject")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def _plot_comparison_bar(results_dict: dict, save_path):
    """Bar chart comparing Spearman ρ across probe conditions."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, means, stds = [], [], []
    for name, res in results_dict.items():
        if "aggregate" not in res:
            continue
        labels.append(name)
        means.append(res["aggregate"]["spearman"]["mean"])
        stds.append(res["aggregate"]["spearman"]["std"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=["#d62728" if m < 0 else "#1f77b4" for m in means])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("LOSO Spearman ρ (mean ± std across subjects)")
    ax.set_title("Linear probe comparison")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ── Public API ────────────────────────────────────────────────────────────────

def run_linear_probe(
    data_dir: str,
    pretrained_weights: Optional[str] = None,
    output_dir: str = "results/linearprobe",
    ridge_alpha: float = 1.0,
    device: str = "auto",
    artifacts_dir: str = "artifacts",
) -> dict:
    """
    Run the full linear probe evaluation suite on UDysRS.

    Evaluations performed (in order):
      - normalization_audit:  signal preservation check before LOSO
      - kinematic_baseline:   LOSO on 17 handcrafted features (speed/symmetry/jerk/ROM)
      - combined:             LOSO on encoder embeddings, all tasks together
      - per_task:             separate LOSO for drinking / communication / la

    Architecture config is read from the checkpoint; no need to specify manually.
    For random-init baseline, omit pretrained_weights (defaults to 128/4/4/60).

    Returns
    -------
    dict with keys: normalization_audit, kinematic_baseline, combined, per_task,
                    and a top-level "config" entry.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    encoder, seq_len = _load_encoder(pretrained_weights, dev)

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_udysrs(data_dir)
    arrays      = data["arrays"]
    scores_raw  = data["scores_raw"]
    tasks       = data["task"]
    subject_ids = data["subject_ids"]
    N = len(arrays)
    print(f"\nLoaded {N} clips from {len(set(subject_ids))} subjects")

    # ── 1. Normalization audit ─────────────────────────────────────────────────
    audit = normalization_audit(arrays, scores_raw, tasks)

    # ── 2. Kinematic feature baseline ─────────────────────────────────────────
    print("\n=== Kinematic Feature Baseline ===")
    print("Computing 17 handcrafted kinematic features (speed/asymmetry/coordination/jerk/ROM)...")
    X_kin = _compute_kinematic_features(arrays)           # (N, 17)
    X_kin = StandardScaler().fit_transform(X_kin)         # z-score features
    kin_results = _run_loso(X_kin, scores_raw, tasks, subject_ids, ridge_alpha,
                            label="kinematic_baseline")

    # ── 3. Encoder embeddings — combined ──────────────────────────────────────
    print(f"\n=== Encoder Probe (combined) ===")
    print("Encoding all clips (frozen encoder)...")
    X_enc = _encode_clips(arrays, encoder, seq_len, dev)  # (N, embed_dim)
    enc_results = _run_loso(X_enc, scores_raw, tasks, subject_ids, ridge_alpha,
                            label="encoder_combined")

    # ── 4. Per-task probes ────────────────────────────────────────────────────
    print("\n=== Per-Task Encoder Probes ===")
    per_task_results = {}
    for task in ["drinking", "communication", "la"]:
        idx  = [i for i, t in enumerate(tasks) if t == task]
        if len(idx) < 10:
            continue
        X_t  = X_enc[idx]
        s_t  = scores_raw[idx]
        ta_t = [tasks[i] for i in idx]
        su_t = [subject_ids[i] for i in idx]
        per_task_results[task] = _run_loso(X_t, s_t, ta_t, su_t, ridge_alpha,
                                           label=f"encoder_{task}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = pathlib.Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    yt  = np.array(enc_results["all_y_true"])
    yp  = np.array(enc_results["all_y_pred"])
    _plot_scatter(yt, yp, enc_results["all_tasks_test"],
                  out_dir / "scatter_combined.png",
                  title=f"Encoder LOSO  ρ={enc_results['aggregate']['spearman']['mean']:+.3f}")
    _plot_per_task(yt, yp, enc_results["all_tasks_test"],
                   out_dir / "scatter_per_task.png")
    _plot_spearman_by_subject(enc_results["fold_results"],
                              out_dir / "spearman_by_subject.png")

    # Comparison bar chart across all conditions
    comparison = {
        "kinematic_baseline": kin_results,
        "encoder_combined":   enc_results,
        **{f"encoder_{t}": r for t, r in per_task_results.items()},
    }
    _plot_comparison_bar(comparison, out_dir / "probe_comparison.png")

    metrics = {
        "normalization_audit": audit,
        "kinematic_baseline":  {k: v for k, v in kin_results.items()
                                 if k in ("aggregate", "per_task", "fold_results")},
        "combined":            {k: v for k, v in enc_results.items()
                                 if k in ("aggregate", "per_task", "fold_results")},
        "per_task_probes":     {t: {k: v for k, v in r.items()
                                    if k in ("aggregate", "per_task", "fold_results")}
                                for t, r in per_task_results.items()},
        "config": {
            "pretrained_weights": str(pretrained_weights),
            "ridge_alpha":        ridge_alpha,
            "seq_len":            seq_len,
        },
    }
    with open(out_dir / "loso_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"  loso_metrics.json  scatter_combined.png  probe_comparison.png")
    return metrics
