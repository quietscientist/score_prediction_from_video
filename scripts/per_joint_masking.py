"""
Per-joint occlusion sensitivity analysis for the pretrained Pose-JEPA encoder.

For each of the 17 COCO joints, all frames of that joint are zeroed out and the
resulting CLS embedding is compared to the unmasked embedding via L2 distance.
Large shift = the encoder is sensitive to that joint = it carries movement information.

Outputs (saved to --output-dir):
  joint_sensitivity.png   — skeleton heatmap, one column per group (normal / abnormal)
                            + a difference map (abnormal − normal)
  joint_sensitivity.json  — raw per-joint mean L2 values per group

Usage:
  python scripts/per_joint_masking.py \\
    --data-dir /path/to/gma/pose_json \\
    --scores-file /path/to/gma_scores.csv \\
    --checkpoint /path/to/checkpoint_ep0010.pt \\
    --output-dir results/per_joint_masking
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from kinescope.linearprobe.evaluate import _load_encoder
from kinescope.linearprobe.gma_loader import load_gma
from kinescope.pretrain._normalize_clip import normalize_clip
from kinescope.skeleton import COCO_LIMBS, COCO_PART_NAMES

N_JOINTS = 17


# ── Occlusion sensitivity ──────────────────────────────────────────────────────

def compute_joint_sensitivity(
    arrays: list,
    encoder,
    seq_len: int,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    For each clip and each joint, compute L2 distance between:
      - CLS embedding of full clip
      - CLS embedding of clip with that joint zeroed across all frames

    Returns (N, 17) float32 array of per-clip per-joint sensitivity scores.
    """
    import torch

    encoder.eval()
    stride = seq_len // 2
    N = len(arrays)
    sensitivities = np.zeros((N, N_JOINTS), dtype=np.float32)

    with torch.no_grad():
        for i, arr in enumerate(arrays):
            if i % 100 == 0:
                print(f"  Processing clip {i}/{N} ...", flush=True)

            normalized = normalize_clip(np.nan_to_num(arr, nan=0.0))
            T = len(normalized)

            # Chunk into windows
            starts = list(range(0, T - seq_len + 1, stride))
            if not starts:
                padded = np.zeros((seq_len, 17, 2), dtype=np.float32)
                padded[:T] = normalized
                chunks = [padded]
            else:
                chunks = [normalized[s:s + seq_len] for s in starts]

            clips_np = np.stack(chunks)  # (C, seq_len, 17, 2)
            clips = torch.tensor(clips_np, dtype=torch.float32).to(device)

            # Full-clip CLS embedding: mean over chunks
            full_emb = encoder(clips).mean(0)  # (embed_dim,)

            # Per-joint masked embeddings
            for j in range(N_JOINTS):
                clips_masked = clips.clone()
                clips_masked[:, :, j, :] = 0.0          # zero joint j all frames
                masked_emb = encoder(clips_masked).mean(0)  # (embed_dim,)
                sensitivities[i, j] = (full_emb - masked_emb).norm().item()

    return sensitivities  # (N, 17)


# ── Skeleton heatmap plot ──────────────────────────────────────────────────────

def _skeleton_layout():
    """
    Approximate 2D positions for each COCO-17 joint in a canonical upright pose.
    Returns (17, 2) array with x in [-1, 1], y in [-1, 1] (y up).
    """
    pos = np.zeros((17, 2), dtype=np.float32)
    # Face
    pos[0]  = [ 0.00,  1.00]  # nose
    pos[1]  = [-0.10,  1.05]  # left_eye
    pos[2]  = [ 0.10,  1.05]  # right_eye
    pos[3]  = [-0.20,  1.00]  # left_ear
    pos[4]  = [ 0.20,  1.00]  # right_ear
    # Upper body
    pos[5]  = [-0.35,  0.65]  # left_shoulder
    pos[6]  = [ 0.35,  0.65]  # right_shoulder
    pos[7]  = [-0.55,  0.30]  # left_elbow
    pos[8]  = [ 0.55,  0.30]  # right_elbow
    pos[9]  = [-0.65,  0.00]  # left_wrist
    pos[10] = [ 0.65,  0.00]  # right_wrist
    # Lower body
    pos[11] = [-0.20,  0.10]  # left_hip
    pos[12] = [ 0.20,  0.10]  # right_hip
    pos[13] = [-0.22, -0.35]  # left_knee
    pos[14] = [ 0.22, -0.35]  # right_knee
    pos[15] = [-0.22, -0.75]  # left_ankle
    pos[16] = [ 0.22, -0.75]  # right_ankle
    return pos


def plot_skeleton_heatmap(
    sensitivity_per_group: dict,
    save_path: pathlib.Path,
):
    """
    Draw skeleton heatmaps side by side, one per group, plus a difference map.
    Joint color = sensitivity value (cool-warm colormap).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    pos = _skeleton_layout()
    groups = list(sensitivity_per_group.keys())
    n_panels = len(groups) + 1  # groups + difference
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))

    # Compute difference: last group − first group (abnormal − normal if available)
    vals_list = [sensitivity_per_group[g] for g in groups]
    diff = vals_list[-1] - vals_list[0] if len(vals_list) >= 2 else vals_list[0]

    all_vals = np.concatenate(vals_list)
    vmin, vmax = all_vals.min(), all_vals.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd

    diff_abs = np.abs(diff).max()
    diff_norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-diff_abs, vmax=diff_abs)
    diff_cmap = plt.cm.RdBu_r

    def _draw_skeleton(ax, values, norm, cmap, title):
        # Draw limbs
        for (a, b) in COCO_LIMBS:
            ax.plot(
                [pos[a, 0], pos[b, 0]],
                [pos[a, 1], pos[b, 1]],
                color="#cccccc", lw=1.5, zorder=1,
            )
        # Draw joints
        colors = cmap(norm(values))
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap=cmap, norm=norm,
                        s=300, zorder=2, edgecolors="white", linewidths=0.8)
        # Label joint names (short)
        short_names = [n.replace("left_", "L.").replace("right_", "R.")
                       for n in COCO_PART_NAMES]
        for j, (x, y) in enumerate(pos):
            ax.annotate(short_names[j], (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=5.5, color="#444444")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-1.0, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return sc

    for ax, group_name, vals in zip(axes[:-1], groups, vals_list):
        sc = _draw_skeleton(ax, vals, norm, cmap, group_name)
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     fraction=0.03, label="L2 sensitivity")

    # Difference panel
    _draw_skeleton(axes[-1], diff, diff_norm, diff_cmap,
                   f"Difference\n({groups[-1]} − {groups[0]})" if len(groups) >= 2 else "Sensitivity")
    plt.colorbar(ScalarMappable(norm=diff_norm, cmap=diff_cmap), ax=axes[-1],
                 fraction=0.03, label="ΔL2")

    fig.suptitle(
        "Per-Joint Encoder Sensitivity (occlusion)\n"
        "Higher = encoder relies more on that joint",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",    required=True,
                        help="Directory of GMA pose JSON files")
    parser.add_argument("--scores-file", default=None,
                        help="Path to gma_scores.csv")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to pretrained encoder checkpoint (.pt)")
    parser.add_argument("--output-dir",  default="results/per_joint_masking")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--max-clips",   type=int, default=None,
                        help="Cap clips for speed (default: all)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dev = torch.device(device)

    print(f"Loading encoder from {args.checkpoint} ...")
    encoder, seq_len = _load_encoder(args.checkpoint, dev)

    print("Loading GMA data ...")
    data = load_gma(args.data_dir, scores_file=args.scores_file)
    arrays   = data["arrays"]
    y_binary = data["binary"]
    print(f"  {len(arrays)} clips  "
          f"(normal={int((y_binary==0).sum())}  abnormal={int((y_binary==1).sum())})")

    if args.max_clips is not None and len(arrays) > args.max_clips:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(arrays), size=args.max_clips, replace=False)
        arrays   = [arrays[i] for i in idx]
        y_binary = y_binary[idx]
        print(f"  Subsampled to {args.max_clips} clips")

    print("Computing per-joint sensitivity ...")
    sens = compute_joint_sensitivity(arrays, encoder, seq_len, dev)  # (N, 17)

    normal_mask   = y_binary == 0
    abnormal_mask = y_binary == 1
    sensitivity_per_group = {
        f"Normal F+\n(n={normal_mask.sum()})":      sens[normal_mask].mean(0),
        f"Abnormal F+/-/F-\n(n={abnormal_mask.sum()})": sens[abnormal_mask].mean(0),
    }

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting ...")
    plot_skeleton_heatmap(sensitivity_per_group, out_dir / "joint_sensitivity.png")

    results = {
        "joints": COCO_PART_NAMES,
        "groups": {
            name.replace("\n", " "): vals.tolist()
            for name, vals in sensitivity_per_group.items()
        },
        "checkpoint": args.checkpoint,
    }
    with open(out_dir / "joint_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"  joint_sensitivity.png  joint_sensitivity.json")

    # Print table
    print(f"\n{'Joint':<18}", end="")
    for name in sensitivity_per_group:
        print(f"  {name.split(chr(10))[0]:>18}", end="")
    print()
    for j, jname in enumerate(COCO_PART_NAMES):
        print(f"  {jname:<16}", end="")
        for vals in sensitivity_per_group.values():
            print(f"  {vals[j]:>18.4f}", end="")
        print()


if __name__ == "__main__":
    main()
