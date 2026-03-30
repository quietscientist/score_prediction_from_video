"""
Stratified k-fold linear probe evaluation on the GMA dataset.

Uses a multiple-instance learning (MIL) setup:
  - All sessions per infant are used (not just the last).
  - Each seq_len window of a recording is a separate training example.
  - Splits are subject-level: no infant spans train and test folds.
  - Test AUROC is computed on per-infant aggregated predictions
    (mean window score per infant), not per-window.

This avoids the information loss of mean-pooling a full recording into
one embedding, and increases training signal from ~900 to ~N*W examples
where W is the mean number of windows per recording.

Runs two evaluations:
  1. Kinematic baseline — 38 handcrafted features per window
  2. Encoder probe     — frozen PoseJEPA CLS embedding per window

Primary metric: AUROC (subject-aggregated).
Also reports balanced accuracy and MCC on the subject-aggregated predictions.

Usage
-----
from kinescope.linearprobe.evaluate_gma import run_gma_probe
results = run_gma_probe(
    data_dir="/path/to/gma/",
    pretrained_weights="./checkpoints/best.pt",
    output_dir="./results/gma_probe",
)
print(results["encoder"]["pooled_auroc"])
"""

import json
import pathlib
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from kinescope.linearprobe.evaluate import (
    _compute_kinematic_features,
    _load_encoder,
)
from kinescope.linearprobe.gma_loader import load_gma
from kinescope.pretrain._normalize_clip import normalize_clip


# ── Window-level feature extraction ───────────────────────────────────────────

def _windows_from_array(arr: np.ndarray, seq_len: int) -> list:
    """
    Normalize and chunk a (T, 17, 2) array into seq_len windows.
    Returns list of (seq_len, 17, 2) arrays. Short clips are zero-padded.
    """
    normalized = normalize_clip(np.nan_to_num(arr, nan=0.0))
    T = len(normalized)
    stride = seq_len // 2
    starts = list(range(0, T - seq_len + 1, stride))
    if not starts:
        padded = np.zeros((seq_len, 17, 2), dtype=np.float32)
        padded[:T] = normalized
        return [padded]
    return [normalized[s:s + seq_len] for s in starts]


def _encode_windows(arrays: list, subject_ids: list, y_binary: np.ndarray,
                    encoder, seq_len: int, device) -> tuple:
    """
    Expand each recording into per-window CLS embeddings.

    Returns
    -------
    X       : (N_windows, embed_dim) float32
    y       : (N_windows,) int32  — label of the parent recording
    groups  : (N_windows,) str array — infant_id of each window
    """
    import torch

    encoder.eval()
    X_list, y_list, g_list = [], [], []

    with torch.no_grad():
        for arr, sid, label in zip(arrays, subject_ids, y_binary):
            windows = _windows_from_array(arr, seq_len)
            batch = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)
            embs = encoder(batch).cpu().numpy()   # (W, embed_dim)
            X_list.append(embs)
            y_list.extend([label] * len(windows))
            g_list.extend([sid] * len(windows))

    return (np.concatenate(X_list, axis=0).astype(np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(g_list))


def _kinematic_windows(arrays: list, subject_ids: list, y_binary: np.ndarray,
                       seq_len: int) -> tuple:
    """
    Expand each recording into per-window kinematic features (38-dim).
    """
    X_list, y_list, g_list = [], [], []
    for arr, sid, label in zip(arrays, subject_ids, y_binary):
        windows = _windows_from_array(arr, seq_len)
        feats = _compute_kinematic_features(windows)   # (W, 38)
        X_list.append(feats)
        y_list.extend([label] * len(windows))
        g_list.extend([sid] * len(windows))

    return (np.concatenate(X_list, axis=0).astype(np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(g_list))


# ── Subject-level k-fold loop ─────────────────────────────────────────────────

def _run_subject_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    label: str = "",
) -> dict:
    """
    StratifiedGroupKFold: stratified on binary label, no subject spans folds.
    Train on window-level features; aggregate predictions per subject for AUROC.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    all_subj_true, all_subj_score = [], []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]
        g_test  = groups[test_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000, random_state=42,
        )
        clf.fit(X_train, y_train)
        y_score_win = clf.predict_proba(X_test)[:, 1]

        # Aggregate per subject: mean predicted score, ground-truth label
        unique_subjs = np.unique(g_test)
        subj_true, subj_score = [], []
        for subj in unique_subjs:
            mask = g_test == subj
            subj_true.append(int(y_test[mask][0]))        # all windows same label
            subj_score.append(float(y_score_win[mask].mean()))

        subj_true  = np.array(subj_true)
        subj_score = np.array(subj_score)

        auroc = (float(roc_auc_score(subj_true, subj_score))
                 if 0 < subj_true.sum() < len(subj_true) else float("nan"))
        y_pred = (subj_score >= 0.5).astype(int)
        bacc   = float(balanced_accuracy_score(subj_true, y_pred))
        mcc    = float(matthews_corrcoef(subj_true, y_pred))

        fold_results.append(dict(fold=fold, n_subjects=len(unique_subjs),
                                 auroc=auroc, balanced_acc=bacc, mcc=mcc))
        all_subj_true.extend(subj_true.tolist())
        all_subj_score.extend(subj_score.tolist())

    def _agg(key):
        vals = [f[key] for f in fold_results if np.isfinite(f[key])]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals \
               else {"mean": float("nan"), "std": float("nan")}

    aggregate = {k: _agg(k) for k in ("auroc", "balanced_acc", "mcc")}
    aggregate["n_folds"] = len(fold_results)

    yt_all = np.array(all_subj_true)
    ys_all = np.array(all_subj_score)
    pooled_auroc = (float(roc_auc_score(yt_all, ys_all))
                    if 0 < yt_all.sum() < len(yt_all) else float("nan"))

    hdr = f"[{label}] " if label else ""
    agg = aggregate
    print(f"\n{hdr}Aggregate ({n_splits}-fold subject-level, "
          f"n_subjects={len(np.unique(groups))})")
    print(f"  AUROC={agg['auroc']['mean']:.3f}±{agg['auroc']['std']:.3f}"
          f"  (pooled={pooled_auroc:.3f})"
          f"  BalAcc={agg['balanced_acc']['mean']:.3f}±{agg['balanced_acc']['std']:.3f}"
          f"  MCC={agg['mcc']['mean']:+.3f}±{agg['mcc']['std']:.3f}")

    return dict(
        fold_results  = fold_results,
        aggregate     = aggregate,
        pooled_auroc  = pooled_auroc,
        all_y_true    = all_subj_true,
        all_y_score   = all_subj_score,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_roc(kin_res: dict, enc_res: dict, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    for res, color, label in [
        (kin_res, "#8C8C8C", "Kinematic (38 features)"),
        (enc_res, "#4C72B0", "Encoder (frozen)"),
    ]:
        yt = np.array(res["all_y_true"])
        ys = np.array(res["all_y_score"])
        if 0 < yt.sum() < len(yt):
            fpr, tpr, _ = roc_curve(yt, ys)
            auc = res["pooled_auroc"]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}  AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("GMA Probe — Abnormal Fidgety Detection\n"
                 "(pooled 5-fold subject-level MIL, score≥2 = abnormal)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_score_dist(y_binary: np.ndarray, scores_raw: np.ndarray, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dist = Counter(scores_raw.tolist())
    labels = sorted(dist.keys())
    counts = [dist[k] for k in labels]
    score_names = {1: "F+\n(normal)", 2: "F+/-\n(sporadic)", 3: "F-\n(absent)"}
    axes[0].bar([score_names.get(l, str(l)) for l in labels], counts,
                color=["#2ca02c", "#ff7f0e", "#d62728"])
    axes[0].set_title("Score distribution (3-class)")
    axes[0].set_ylabel("Count")
    for i, (l, c) in enumerate(zip(labels, counts)):
        axes[0].text(i, c + 1, str(c), ha="center", fontsize=9)

    n_normal   = int((y_binary == 0).sum())
    n_abnormal = int((y_binary == 1).sum())
    axes[1].bar(["Normal (F+)", "Abnormal (F+/- or F-)"], [n_normal, n_abnormal],
                color=["#2ca02c", "#d62728"])
    axes[1].set_title("Binary label (score≥2 = abnormal)")
    axes[1].set_ylabel("Count")
    for i, c in enumerate([n_normal, n_abnormal]):
        axes[1].text(i, c + 1, str(c), ha="center", fontsize=9)

    fig.suptitle("GMA Dataset — Label Distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_gma_probe(
    data_dir: str,
    scores_file: Optional[str] = None,
    pretrained_weights: Optional[str] = None,
    output_dir: str = "results/gma_probe",
    n_splits: int = 5,
    C: float = 1.0,
    device: str = "auto",
    skip_kinematic: bool = False,
) -> dict:
    """
    Run the GMA linear probe evaluation (MIL setup).

    Parameters
    ----------
    data_dir : str
        GMA dataset directory containing pose JSON files and gma_scores.csv.
    pretrained_weights : str, optional
        Path to PoseJEPA checkpoint. None = random-init baseline.
    output_dir : str
        Where to save metrics JSON, ROC plot, and predictions.
    n_splits : int
        StratifiedGroupKFold splits (default 5).
    C : float
        Logistic regression regularization (default 1.0).
    device : str
        'cuda', 'cpu', or 'auto'.

    Returns
    -------
    dict with keys: kinematic, encoder, config.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    encoder, seq_len = _load_encoder(pretrained_weights, dev)

    # ── Load data — all sessions ───────────────────────────────────────────────
    data = load_gma(data_dir, scores_file=scores_file, all_sessions=True)
    arrays      = data["arrays"]
    y_binary    = data["binary"]
    scores_raw  = data["scores_raw"]
    subject_ids = data["subject_ids"]
    N = len(arrays)
    print(f"\nLoaded {N} recordings  "
          f"(normal={int((y_binary==0).sum())}  abnormal={int((y_binary==1).sum())})")

    # ── Kinematic baseline ─────────────────────────────────────────────────────
    if skip_kinematic:
        print("\n=== Kinematic Feature Baseline: SKIPPED (--skip-kinematic) ===")
        kin_results = {"pooled_auroc": float("nan"), "all_y_true": [], "all_y_score": [],
                       "aggregate": {}, "fold_results": []}
    else:
        print("\n=== Kinematic Feature Baseline (38 features, MIL) ===")
        X_kin, y_kin, g_kin = _kinematic_windows(arrays, subject_ids, y_binary, seq_len)
        print(f"  {len(X_kin)} windows from {len(arrays)} recordings  "
              f"(mean {len(X_kin)/len(arrays):.1f} windows/recording)")
        kin_results = _run_subject_kfold(X_kin, y_kin, g_kin,
                                         n_splits=n_splits, C=C,
                                         label="kinematic_baseline")

    # ── Encoder probe ──────────────────────────────────────────────────────────
    print(f"\n=== Encoder Probe (frozen, {seq_len}-frame windows, MIL) ===")
    X_enc, y_enc, g_enc = _encode_windows(arrays, subject_ids, y_binary,
                                           encoder, seq_len, dev)
    print(f"  {len(X_enc)} windows from {len(arrays)} recordings")
    enc_results = _run_subject_kfold(X_enc, y_enc, g_enc,
                                     n_splits=n_splits, C=C,
                                     label="encoder")

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_roc(kin_results, enc_results, out_dir / "roc_curve.png")
    _plot_score_dist(y_binary, scores_raw, out_dir / "score_distribution.png")

    metrics = {
        "kinematic": {k: v for k, v in kin_results.items()
                      if k in ("aggregate", "pooled_auroc", "fold_results")},
        "encoder":   {k: v for k, v in enc_results.items()
                      if k in ("aggregate", "pooled_auroc", "fold_results")},
        "config": {
            "pretrained_weights": str(pretrained_weights),
            "n_splits":           n_splits,
            "C":                  C,
            "seq_len":            seq_len,
            "n_recordings":       N,
            "n_windows_kin":      len(X_kin) if not skip_kinematic else None,
            "n_windows_enc":      len(X_enc),
            "n_normal":           int((y_binary == 0).sum()),
            "n_abnormal":         int((y_binary == 1).sum()),
            "mil":                True,
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    preds = {
        "kinematic": {"y_true": kin_results["all_y_true"],
                      "y_score": kin_results["all_y_score"]},
        "encoder":   {"y_true": enc_results["all_y_true"],
                      "y_score": enc_results["all_y_score"]},
    }
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(preds, f)

    print(f"\nResults saved to {out_dir}/")
    print(f"  metrics.json  roc_curve.png  score_distribution.png  predictions.json")
    return metrics
