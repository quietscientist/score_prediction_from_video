"""
Penn Action loader: MATLAB label files (.mat) -> normalized COCO-17 clips.

Directory layout expected:
    Penn_Action/
    ├── frames/
    │   ├── 0001/000001.jpg ...
    │   └── ...
    └── labels/
        ├── 0001.mat
        └── ...

Each label file contains:
    x, y       : (T, 13) keypoint coordinates
    visibility : (T, 13) 0/1 visibility flags
    nframes    : scalar
    action     : scalar string

Penn keypoint order (13):
    0  head
    1  left_shoulder
    2  right_shoulder
    3  left_elbow
    4  right_elbow
    5  left_wrist
    6  right_wrist
    7  left_hip
    8  right_hip
    9  left_knee
    10 right_knee
    11 left_ankle
    12 right_ankle
"""

import pathlib

import numpy as np

from kinescope.pretrain._normalize_clip import normalize_clip
from kinescope.skeleton import COCO_PART_NAMES

N_JOINTS = len(COCO_PART_NAMES)  # 17

# Penn-13 -> COCO-17 index
_PENN_TO_COCO = {
    0: 0,    # head -> nose (approximate)
    1: 5,    # left_shoulder
    2: 6,    # right_shoulder
    3: 7,    # left_elbow
    4: 8,    # right_elbow
    5: 9,    # left_wrist
    6: 10,   # right_wrist
    7: 11,   # left_hip
    8: 12,   # right_hip
    9: 13,   # left_knee
    10: 14,  # right_knee
    11: 15,  # left_ankle
    12: 16,  # right_ankle
}


def _interp_nan_1d(values: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs; fallback to zeros when no valid samples exist."""
    values = values.astype(np.float32, copy=True)
    idx = np.arange(len(values))
    valid = np.isfinite(values)
    if valid.sum() == 0:
        values[:] = 0.0
        return values
    if valid.sum() == 1:
        values[:] = values[valid][0]
        return values
    values[~valid] = np.interp(idx[~valid], idx[valid], values[valid]).astype(np.float32)
    return values


def _parse_label_mat(mat_path: pathlib.Path) -> np.ndarray:
    """
    Parse one Penn Action .mat label into COCO-17 sequence.

    Returns
    -------
    (T, 17, 2) float32 array or None on parse failure.
    """
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise ImportError(
            "scipy is required for Penn Action labels (.mat). Install with: pip install scipy"
        ) from exc

    try:
        ann = loadmat(mat_path)
    except Exception:
        return None

    x = ann.get("x")
    y = ann.get("y")
    vis = ann.get("visibility")

    if x is None or y is None or vis is None:
        return None
    if x.ndim != 2 or y.ndim != 2 or vis.ndim != 2:
        return None
    if x.shape != y.shape or x.shape != vis.shape:
        return None
    if x.shape[1] != 13:
        return None

    T = x.shape[0]
    seq = np.full((T, N_JOINTS, 2), np.nan, dtype=np.float32)

    # Copy visible Penn joints into COCO layout
    visible = vis.astype(bool)
    for p_idx, c_idx in _PENN_TO_COCO.items():
        px = x[:, p_idx].astype(np.float32)
        py = y[:, p_idx].astype(np.float32)
        px[~visible[:, p_idx]] = np.nan
        py[~visible[:, p_idx]] = np.nan
        seq[:, c_idx, 0] = px
        seq[:, c_idx, 1] = py

    # Fill missing eye/ear keypoints with nose/head track
    seq[:, 1:5] = seq[:, 0:1]

    # Interpolate NaNs per joint/channel for stable normalization.
    for j in range(N_JOINTS):
        seq[:, j, 0] = _interp_nan_1d(seq[:, j, 0])
        seq[:, j, 1] = _interp_nan_1d(seq[:, j, 1])

    return seq


def load_penn_clips(
    data_dir: pathlib.Path,
    seq_len: int = 60,
    max_files: int = None,
) -> np.ndarray:
    """
    Load Penn Action label files, map Penn-13 -> COCO-17, normalize, and chunk clips.

    Parameters
    ----------
    data_dir : Path - Penn_Action root (contains labels/)
    seq_len : int - clip length in frames
    max_files : int or None - cap number of label files

    Returns
    -------
    (N, seq_len, 17, 2) float32 array
    """
    data_dir = pathlib.Path(data_dir)
    label_dir = data_dir / "labels"
    mat_files = sorted(label_dir.glob("*.mat")) if label_dir.exists() else []

    if not mat_files:
        raise FileNotFoundError(
            f"No Penn Action label files (*.mat) found under {label_dir}."
        )

    if max_files is not None:
        mat_files = mat_files[:max_files]

    all_clips = []
    step = max(1, seq_len // 2)

    for mat_path in mat_files:
        seq = _parse_label_mat(mat_path)
        if seq is None or len(seq) < seq_len:
            continue

        try:
            normalized = normalize_clip(seq)  # (T, 17, 2)
        except Exception:
            continue

        T = normalized.shape[0]
        for start in range(0, T - seq_len + 1, step):
            clip = normalized[start : start + seq_len]
            if np.isfinite(clip).all() and np.abs(clip).max() < 5.0:
                all_clips.append(clip)

    if not all_clips:
        return np.empty((0, seq_len, N_JOINTS, 2), dtype=np.float32)

    return np.stack(all_clips).astype(np.float32)
