"""Visualization utilities for pose clips and Pose-JEPA masking patterns.

Saves plots to artifacts/ for human review of:
  - synthetic_pose_clip.png: stick figure frames from a normalized clip
  - jepa_mask_pattern.png:   same frames with masked joints highlighted in red
  - pretrain_loss_curve.png: JEPA + TPC loss over epochs
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from kinescope.skeleton import COCO_LIMBS, COCO_PART_INDEX, COCO_PART_NAMES, LIMB_JOINTS

LIMB_JOINT_INDICES = set(COCO_PART_INDEX[j] for j in LIMB_JOINTS)
# Face joints (nose, eyes, ears) — skipped in visualization since many datasets omit them
_FACE_JOINTS = frozenset(range(5))
_BODY_LIMBS = [(i, j) for i, j in COCO_LIMBS if i not in _FACE_JOINTS and j not in _FACE_JOINTS]

# Lower body joint indices (11–16): hip, knee, ankle on each side
_LOWER_BODY = frozenset(range(11, 17))


def _normalized_to_display(frame: np.ndarray) -> np.ndarray:
    """
    Convert a normalized COCO-17 frame to a displayable coordinate system.

    After normalize_clip(), the coordinate system is split:
      - Upper body (joints 0–10): y=0 = shoulder center, positive y = physically above
      - Lower body (joints 11–16): y=1.0 = hip center, smaller y = physically below hips

    To unify into a single global y-up frame centred at the shoulder midpoint:
      - Upper body: keep as-is
      - Lower body: subtract 2.0 from y  →  hip becomes (1.0 - 2.0) = -1.0
        (i.e., one trunk-length below shoulder), knee/ankle go further negative

    Plot with matplotlib default (y increases upward, no invert_yaxis).
    """
    disp = frame.copy()
    for idx in _LOWER_BODY:
        disp[idx, 1] -= 2.0
    return disp


def _draw_skeleton(
    ax,
    frame: np.ndarray,
    alpha: float = 1.0,
    limb_color: str = "gray",
    masked_joints: Optional[np.ndarray] = None,
):
    """Draw a single skeleton frame (body joints only). frame: (17, 2) normalized array."""
    frame = _normalized_to_display(frame)
    for i, j in _BODY_LIMBS:
        if np.isnan(frame[i]).any() or np.isnan(frame[j]).any():
            continue
        ax.plot(
            [frame[i, 0], frame[j, 0]],
            [frame[i, 1], frame[j, 1]],
            color=limb_color,
            alpha=alpha,
            linewidth=1.5,
        )

    for idx in range(len(COCO_PART_NAMES)):
        if idx in _FACE_JOINTS:
            continue
        if np.isnan(frame[idx]).any():
            continue
        if masked_joints is not None and masked_joints[idx]:
            ax.scatter(frame[idx, 0], frame[idx, 1], c="red", s=60, zorder=5, alpha=alpha)
        elif idx in LIMB_JOINT_INDICES:
            ax.scatter(
                frame[idx, 0], frame[idx, 1], c="darkorange", s=40, zorder=4, alpha=alpha
            )
        else:
            ax.scatter(
                frame[idx, 0], frame[idx, 1], c="steelblue", s=25, zorder=3, alpha=alpha
            )


def plot_pose_clip(
    clip: np.ndarray,
    save_path: Union[str, Path],
    n_frames: int = 9,
    title: str = "Normalized pose clip",
):
    """
    Plot N evenly-spaced frames from a normalized COCO-17 pose clip as stick figures.

    Parameters
    ----------
    clip : (T, 17, 2) array — normalized pose sequence
    save_path : str or Path
    n_frames : int
    title : str
    """
    T = clip.shape[0]
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    ncols = min(n_frames, 3)
    nrows = (n_frames + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).flatten()

    for k, t in enumerate(frame_indices):
        ax = axes[k]
        _draw_skeleton(ax, clip[t])
        ax.set_title(f"frame {t}", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

    for k in range(n_frames, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_masked_clip(
    clip: np.ndarray,
    mask: np.ndarray,
    save_path: Union[str, Path],
    n_frames: int = 9,
    title: str = "Pose-JEPA masking pattern",
):
    """
    Plot N frames with masked joints highlighted in red.

    Parameters
    ----------
    clip : (T, 17, 2) array
    mask : (T, 17) bool array — True = masked
    save_path : str or Path
    n_frames : int
    title : str
    """
    T = clip.shape[0]
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    ncols = min(n_frames, 3)
    nrows = (n_frames + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).flatten()

    for k, t in enumerate(frame_indices):
        ax = axes[k]
        masked_joints = mask[t]
        _draw_skeleton(ax, clip[t], masked_joints=masked_joints)
        ax.set_title(f"t={t} | masked={masked_joints.sum()}", fontsize=7)
        ax.set_aspect("equal")
        ax.axis("off")

    for k in range(n_frames, len(axes)):
        axes[k].axis("off")

    red_patch = mpatches.Patch(color="red", label="Masked (predict)")
    orange_patch = mpatches.Patch(color="darkorange", label="Limb joint (visible)")
    blue_patch = mpatches.Patch(color="steelblue", label="Other joint (visible)")
    fig.legend(
        handles=[red_patch, orange_patch, blue_patch],
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_loss_curve(
    jepa_losses: list,
    tpc_losses: list,
    save_path: Union[str, Path],
    title: str = "Pretraining loss curve",
):
    """
    Plot Pose-JEPA + motion-gated TPC loss over epochs.

    Parameters
    ----------
    jepa_losses : list of float
    tpc_losses : list of float
    save_path : str or Path
    title : str
    """
    epochs = range(1, len(jepa_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, jepa_losses, label="Pose-JEPA loss", color="steelblue", linewidth=2)
    ax.plot(
        epochs,
        tpc_losses,
        label="TPC loss (motion-gated)",
        color="darkorange",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
