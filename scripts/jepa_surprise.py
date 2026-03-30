"""
Unsupervised movement surprise scoring.

Three modes (in order of preference):

  1. jepa  (default when full checkpoint available)
     Mean MSE between predictor output and target encoder embeddings over
     N random spatiotemporal masks. Directly measures what the model finds
     hard to predict. Requires context_encoder + target_encoder + predictor.

  2. norm  (default when only context_encoder available)
     L2 norm of the CLS embedding. Valid when SIGReg has pushed embeddings
     toward N(0,I) — high-norm clips are statistically unusual movements.
     Simple, fast, avoids curse of dimensionality.

  3. knn   (explicit --mode knn)
     Mean cosine distance to k nearest neighbors in embedding space.
     Useful if embedding distribution is non-Gaussian (SIGReg not used).

No clinical labels are used. Surprise scores are correlated with GMA score
post-hoc to test whether reconstruction difficulty tracks clinical severity.

Outputs (saved to --output-dir):
  surprise_scores.json  — per-clip scores, subject ids, GMA scores
  surprise_vs_score.png — scatter: surprise vs GMA score
  surprise_roc.png      — ROC curve (surprise as unsupervised classifier)
  surprise_dist.png     — score distributions per GMA group

Usage:
  python scripts/jepa_surprise.py \\
    --data-dir /path/to/gma/pose_json \\
    --scores-file /path/to/gma_scores.csv \\
    --checkpoint /path/to/checkpoint.pt \\
    --output-dir results/jepa_surprise
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from kinescope.linearprobe.evaluate import _encode_clips, _load_encoder
from kinescope.linearprobe.gma_loader import load_gma
from kinescope.prediction._vit import PoseJEPA, _sample_block_mask
from kinescope.pretrain._normalize_clip import normalize_clip

N_JOINTS = 17


# ── Mode 1: JEPA reconstruction error ─────────────────────────────────────────

def _load_full_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "predictor" not in ckpt or "target_encoder" not in ckpt:
        raise KeyError("checkpoint missing predictor/target_encoder")

    cfg        = ckpt.get("config", {})
    embed_dim  = cfg.get("embed_dim", 128)
    n_layers   = cfg.get("n_layers",  4)
    n_heads    = cfg.get("n_heads",   4)
    seq_len    = cfg.get("seq_len",   60)
    coord_dim  = cfg.get("coord_dim", 2)

    if not cfg:
        pe        = ckpt["context_encoder"]["token_embedding.temporal_pe.pe"]
        seq_len   = pe.shape[0] - 64
        embed_dim = pe.shape[1]
        n_heads   = max(1, embed_dim // 32)
        n_layers  = sum(1 for k in ckpt["context_encoder"]
                        if k.startswith("transformer.layers.") and
                        k.endswith(".self_attn.in_proj_weight"))

    model = PoseJEPA(
        embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
        seq_len=seq_len, coord_dim=coord_dim,
    ).to(device)
    model.context_encoder.load_state_dict(ckpt["context_encoder"], strict=False)
    model.target_encoder.load_state_dict(ckpt["target_encoder"],   strict=False)
    model.predictor.load_state_dict(ckpt["predictor"],             strict=False)
    model.eval()
    print(f"Loaded full PoseJEPA  (embed={embed_dim}, layers={n_layers}, seq={seq_len})")
    return model, seq_len


def compute_jepa_surprise(arrays, model, seq_len, device, n_masks=8):
    """Mean JEPA MSE over n_masks random masks per clip window."""
    model.eval()
    stride = seq_len // 2
    scores = []
    with torch.no_grad():
        for i, arr in enumerate(arrays):
            if i % 100 == 0:
                print(f"  Clip {i}/{len(arrays)} ...", flush=True)
            normalized = normalize_clip(np.nan_to_num(arr, nan=0.0))
            T = len(normalized)
            starts = list(range(0, T - seq_len + 1, stride)) or [0]
            chunks = []
            for s in starts:
                c = normalized[s:s + seq_len]
                if len(c) < seq_len:
                    pad = np.zeros((seq_len, N_JOINTS, 2), dtype=np.float32)
                    pad[:len(c)] = c
                    c = pad
                chunks.append(c)

            clip_scores = []
            for chunk in chunks:
                x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
                _, T_c, J, _ = x.shape
                target_all = model.target_encoder.encode_tokens(x)[:, 1:].view(1, T_c, J, -1)
                mask_errors = []
                for _ in range(n_masks):
                    mask = _sample_block_mask(T_c, J, mask_ratio=0.5, device=device).unsqueeze(0)
                    x_m = x.clone(); x_m[mask] = 0.0
                    ctx = model.context_encoder.encode_tokens(x_m)[:, 1:].view(1, T_c, J, -1)
                    m = mask[0]
                    if m.sum() == 0 or (~m).sum() == 0:
                        continue
                    t_idx, j_idx = m.nonzero(as_tuple=True)
                    pos_enc = model.context_encoder.token_embedding.position_encoding(t_idx, j_idx, device)
                    pred = model.predictor(ctx[0][~m].unsqueeze(0), pos_enc.unsqueeze(0)).squeeze(0)
                    mask_errors.append(F.mse_loss(pred, target_all[0][m]).item())
                if mask_errors:
                    clip_scores.append(float(np.mean(mask_errors)))
            scores.append(float(np.mean(clip_scores)) if clip_scores else float("nan"))
    return np.array(scores, dtype=np.float32)


# ── Mode 2: L2 norm of CLS embedding ──────────────────────────────────────────

def compute_norm_surprise(arrays, encoder, seq_len, device):
    """
    L2 norm of mean CLS embedding per clip.
    Under SIGReg (N(0,I) target), high norm = statistically unusual movement.
    """
    embs = _encode_clips(arrays, encoder, seq_len, device)   # (N, E)
    return np.linalg.norm(embs, axis=1).astype(np.float32)   # (N,)


# ── Mode 3: k-NN outlier distance ─────────────────────────────────────────────

def compute_knn_surprise(arrays, encoder, seq_len, device, k=10):
    """Mean cosine distance to k nearest neighbors in embedding space."""
    from sklearn.metrics.pairwise import cosine_distances
    embs = _encode_clips(arrays, encoder, seq_len, device)
    dists = cosine_distances(embs)
    np.fill_diagonal(dists, np.inf)
    return np.sort(dists, axis=1)[:, :k].mean(axis=1).astype(np.float32)


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_results(surprise, scores_raw, y_binary, out_dir, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, roc_curve

    finite = np.isfinite(surprise)
    s, sc, yb = surprise[finite], scores_raw[finite], y_binary[finite]
    rho   = float(spearmanr(s, sc).statistic)
    auroc = float(roc_auc_score(yb, s)) if 0 < yb.sum() < len(yb) else float("nan")

    colors = {1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    for sv in [1, 2, 3]:
        m = sc == sv
        ax.scatter(sc[m] + np.random.default_rng(sv).uniform(-0.1, 0.1, m.sum()),
                   s[m], alpha=0.4, s=15, color=colors[sv],
                   label=f"Score {sv} (n={m.sum()})")
    ax.set_xlabel("GMA Score (1=F+, 2=F+/-, 3=F-)")
    ax.set_ylabel("Surprise score")
    ax.set_title(f"Surprise vs GMA Score  ρ={rho:+.3f}\n({mode})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "surprise_vs_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(5, 5))
    if np.isfinite(auroc):
        fpr, tpr, _ = roc_curve(yb, s)
        ax.plot(fpr, tpr, lw=2, color="#4C72B0", label=f"Surprise  AUC={auroc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Unsupervised GMA Detection\n({mode}, zero labels)")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "surprise_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, mask, color in [
        ("Normal F+ (score=1)", yb == 0, "#2ca02c"),
        ("Abnormal (score≥2)",  yb == 1, "#d62728"),
    ]:
        ax.hist(s[mask], bins=40, alpha=0.5, color=color,
                label=f"{label} (n={mask.sum()})", density=True)
    ax.set_xlabel("Surprise score"); ax.set_ylabel("Density")
    ax.set_title(f"Surprise distribution  AUROC={auroc:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "surprise_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSpearman ρ (surprise vs score): {rho:+.3f}")
    print(f"AUROC (unsupervised):           {auroc:.3f}")
    return rho, auroc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--scores-file", default=None)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output-dir",  default="results/jepa_surprise")
    parser.add_argument("--mode",        default="auto",
                        choices=["auto", "jepa", "norm", "knn"],
                        help="Surprise mode: auto selects jepa if full checkpoint "
                             "available, else norm (default: auto)")
    parser.add_argument("--n-masks",     type=int, default=8,
                        help="Mask samples per window for jepa mode (default: 8)")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--max-clips",   type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print("Loading GMA data ...")
    data        = load_gma(args.data_dir, scores_file=args.scores_file)
    arrays      = data["arrays"]
    scores_raw  = data["scores_raw"]
    y_binary    = data["binary"]
    subject_ids = data["subject_ids"]

    if args.max_clips and len(arrays) > args.max_clips:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(arrays), size=args.max_clips, replace=False)
        arrays = [arrays[i] for i in idx]
        scores_raw  = scores_raw[idx]
        y_binary    = y_binary[idx]
        subject_ids = [subject_ids[i] for i in idx]

    print(f"  {len(arrays)} clips  "
          f"(normal={int((y_binary==0).sum())}  abnormal={int((y_binary==1).sum())})")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        # Try full model first; fall back to norm
        try:
            _load_full_model(args.checkpoint, dev)  # probe only
            mode = "jepa"
        except KeyError:
            mode = "norm"
        print(f"Auto-selected mode: {mode}")

    if mode == "jepa":
        model, seq_len = _load_full_model(args.checkpoint, dev)
        print(f"Computing JEPA surprise ({args.n_masks} masks/window) ...")
        surprise = compute_jepa_surprise(arrays, model, seq_len, dev, args.n_masks)
        mode_label = f"JEPA reconstruction error ({args.n_masks} masks)"
    elif mode == "norm":
        encoder, seq_len = _load_encoder(args.checkpoint, dev)
        print("Computing CLS embedding L2 norm ...")
        surprise = compute_norm_surprise(arrays, encoder, seq_len, dev)
        mode_label = "CLS embedding L2 norm (SIGReg-normalized)"
    elif mode == "knn":
        encoder, seq_len = _load_encoder(args.checkpoint, dev)
        print("Computing k-NN outlier score ...")
        surprise = compute_knn_surprise(arrays, encoder, seq_len, dev)
        mode_label = "k-NN cosine outlier distance"

    rho, auroc = _plot_results(surprise, scores_raw, y_binary, out_dir, mode_label)

    with open(out_dir / "surprise_scores.json", "w") as f:
        json.dump({
            "mode":                mode_label,
            "checkpoint":          args.checkpoint,
            "spearman_rho":        rho,
            "auroc_unsupervised":  auroc,
            "n_clips":             len(arrays),
            "n_normal":            int((y_binary == 0).sum()),
            "n_abnormal":          int((y_binary == 1).sum()),
            "subject_ids":         subject_ids,
            "scores_raw":          scores_raw.tolist(),
            "y_binary":            y_binary.tolist(),
            "surprise":            surprise.tolist(),
        }, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
