"""
AMASS dataset loader: SMPL .npz → normalized COCO-17 clips.

Directory layout expected:
    amass/
    ├── CMU/
    │   ├── 01/
    │   │   ├── 01_01_poses.npz
    │   │   └── ...
    ├── MPI_HDM05/
    ├── SFU/
    └── ...  (other AMASS sub-datasets)

Each .npz file contains:
    poses    : (T, 156)  SMPL-H pose parameters (or (T, 72) for SMPL)
    betas    : (16,) or (10,) shape parameters
    trans    : (T, 3) root translation
    gender   : str

Dependencies:
    pip install smplx torch

SMPL → COCO-17 joint mapping
-------------------------------
SMPL has 24 joints (or 52 for SMPL-H). We use the SMPL joint regressor
to extract 3D world positions, then map to the 17 COCO keypoints.

SMPL joint index → COCO-17 index:
    1  (L_Hip)         → 11 left_hip
    2  (R_Hip)         → 12 right_hip
    4  (L_Knee)        → 13 left_knee
    5  (R_Knee)        → 14 right_knee
    7  (L_Ankle)       → 15 left_ankle
    8  (R_Ankle)       → 16 right_ankle
    16 (L_Shoulder)    → 5  left_shoulder
    17 (R_Shoulder)    → 6  right_shoulder
    18 (L_Elbow)       → 7  left_elbow
    19 (R_Elbow)       → 8  right_elbow
    20 (L_Wrist)       → 9  left_wrist
    21 (R_Wrist)       → 10 right_wrist
    15 (Head)          → 0  nose  (approximate)
    -- not present     → 1-4 (eyes, ears): set to NaN
"""

import pathlib

import numpy as np

from kinescope.pretrain._normalize_clip import normalize_clip
from kinescope.skeleton import COCO_PART_NAMES

N_JOINTS = len(COCO_PART_NAMES)  # 17

# SMPL-24 → COCO-17 mapping is handled by rekinect.joints.mapping.smpl24_to_coco17()
# See /home/msegado/tapedeck/msegado/rekinect-pose/rekinect/joints/mapping.py


def _find_model_file(model_root: pathlib.Path, gender: str, pose_dim: int) -> pathlib.Path:
    """
    Locate the SMPL/SMPL-H model npz for a given gender.

    Supports two layouts:
      - smplx-style:  {root}/smplh/{gender}/model.npz   (human_body_prior layout)
      - classic-style: {root}/smplh/SMPLH_{GENDER}.npz
    """
    gender = gender if gender in ("male", "female") else "neutral"
    model_type = "smplh" if pose_dim > 72 else "smpl"

    candidates = [
        model_root / model_type / gender / "model.npz",
        model_root / model_type / f"{model_type.upper()}_{gender.upper()}.npz",
        model_root / model_type / f"{model_type.upper()}_{gender.upper()}.pkl",
        model_root / f"{model_type.upper()}_{gender.upper()}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No {model_type} model file found for gender='{gender}' under {model_root}.\n"
        f"Tried: {[str(c) for c in candidates]}\n"
        f"Set $SMPL_MODELS to the directory containing smplh/ or smpl/ subdirectories."
    )


def _smpl_forward_kinematics(poses: np.ndarray, betas: np.ndarray, gender: str = "neutral"):
    """
    Run SMPL/SMPL-H forward kinematics using human_body_prior.BodyModel.

    Parameters
    ----------
    poses  : (T, 72) or (T, 156) SMPL/SMPL-H pose parameters
    betas  : (10,) or (16,) shape parameters
    gender : 'neutral', 'male', or 'female'

    Returns
    -------
    (T, 24, 3) float32 — world-space joint positions
    """
    import os
    model_root = pathlib.Path(os.environ.get("SMPL_MODELS", str(pathlib.Path.home() / "smpl_models")))

    T = poses.shape[0]
    pose_dim = poses.shape[1]

    bm_path = _find_model_file(model_root, gender, pose_dim)

    joints = _smpl_fk_numpy(bm_path, poses, betas)
    return joints.astype(np.float32)


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """Batch axis-angle (N, 3) → rotation matrices (N, 3, 3)."""
    theta = np.linalg.norm(r, axis=-1, keepdims=True).clip(1e-8)
    k = r / theta  # unit axes (N, 3)
    s, c = np.sin(theta)[..., 0], np.cos(theta)[..., 0]  # (N,)
    # Skew-symmetric matrices
    K = np.zeros((*r.shape[:-1], 3, 3))
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] =  k[..., 0]
    I = np.eye(3)
    R = I + s[..., None, None] * K + (1 - c)[..., None, None] * (K @ K)
    return R


def _smpl_fk_numpy(bm_path: pathlib.Path, poses: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """
    Pure-numpy SMPL/SMPL-H forward kinematics.

    Returns (T, 24, 3) world-space joint positions.
    """
    m = np.load(bm_path, allow_pickle=True)
    J_reg = m["J_regressor_prior"][:24].astype(np.float64)   # (24, 6890)
    v_template = m["v_template"].astype(np.float64)           # (6890, 3)
    shapedirs = m["shapedirs"].astype(np.float64)             # (6890, 3, n_betas)
    kintree = m["kintree_table"].astype(int)                  # (2, 52)

    n_betas = min(len(betas), shapedirs.shape[2])
    b = betas[:n_betas].astype(np.float64)

    # Rest-pose shaped vertices and joints
    v_shaped = v_template + np.einsum("vcd,d->vc", shapedirs[:, :, :n_betas], b)
    J_rest = J_reg @ v_shaped  # (24, 3)

    # Parent array for 24 body joints
    parents = kintree[0, :24].copy()
    parents[0] = -1  # root has no parent

    T = poses.shape[0]
    body_aa = poses[:, :72].reshape(T, 24, 3)  # axis-angle per joint per frame

    # Batch rodrigues: (T*24, 3) → (T*24, 3, 3) → (T, 24, 3, 3)
    R_all = _rodrigues(body_aa.reshape(-1, 3)).reshape(T, 24, 3, 3)

    # Per-joint local-to-parent offset in rest pose
    offsets = J_rest.copy()  # (24, 3)
    for i in range(1, 24):
        offsets[i] = J_rest[i] - J_rest[parents[i]]

    # Forward kinematics: build world transforms (T, 24, 4, 4)
    G = np.zeros((T, 24, 4, 4))
    G[:, :, 3, 3] = 1.0
    for i in range(24):
        R = R_all[:, i]  # (T, 3, 3)
        t = offsets[i]   # (3,)
        G[:, i, :3, :3] = R
        G[:, i, :3, 3] = t
        if parents[i] >= 0:
            G[:, i] = G[:, parents[i]] @ G[:, i]

    joints = G[:, :, :3, 3]  # (T, 24, 3) world positions

    return joints.astype(np.float32)


def _smpl_joints_to_coco(joints: np.ndarray) -> np.ndarray:
    """
    Map SMPL (T, 24, 3) joint positions to COCO-17 (T, 17, 2).

    AMASS stores poses in SMPL world frame: X=right, Y=depth, Z=up.
    Image projection: image_x = SMPL_x, image_y = -SMPL_z (Z=up → flip for image +y=down).

    Uses rekinect smpl24_to_coco17() for the joint index mapping (includes face offsets).

    Returns (T, 17, 2) float32
    """
    import sys as _sys
    _rekinect = "/home/msegado/tapedeck/msegado/rekinect-pose"
    if _rekinect not in _sys.path:
        _sys.path.insert(0, _rekinect)
    from rekinect.joints.mapping import smpl24_to_coco17

    coco_xyz = smpl24_to_coco17(joints.astype(np.float64))  # (T, 17, 3)
    # AMASS world frame: Z=up → image Y = -Z (down is positive in image)
    coco_2d = np.stack([coco_xyz[:, :, 0], -coco_xyz[:, :, 2]], axis=-1)  # (T, 17, 2)
    coco_2d = coco_2d.astype(np.float32)
    # Harmonize face keypoints: copy nose (SMPL head → index 0) to eyes/ears (1-4)
    # so all datasets have a single central head point rather than approximate offsets
    coco_2d[:, 1:5] = coco_2d[:, 0:1]
    return coco_2d


def load_amass_clips(
    data_dir: pathlib.Path,
    seq_len: int = 60,
    max_files: int = None,
) -> np.ndarray:
    """
    Load AMASS .npz files, run SMPL FK, convert to COCO-17, normalize, and chunk.

    Parameters
    ----------
    data_dir : Path — AMASS root directory (contains sub-dataset dirs)
    seq_len : int — clip length in frames
    max_files : int or None — limit number of .npz files (useful for quick testing)

    Returns
    -------
    (N, seq_len, 17, 2) float32 array
    """
    data_dir = pathlib.Path(data_dir)
    npz_files = sorted(data_dir.rglob("*_poses.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"No AMASS pose files (*_poses.npz) found in {data_dir}.\n"
            f"Download from https://amass.is.tue.mpg.de and extract into {data_dir}"
        )

    if max_files is not None:
        npz_files = npz_files[:max_files]

    all_clips = []
    step = seq_len // 2

    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue

        poses = data.get("poses")
        betas = data.get("betas", np.zeros(10))
        gender = str(data.get("gender", "neutral"))

        if poses is None or len(poses) < seq_len:
            continue

        try:
            joints = _smpl_forward_kinematics(poses, betas, gender)  # (T, 24, 3)
        except Exception:
            continue

        coco_2d = _smpl_joints_to_coco(joints)  # (T, 17, 2)
        normalized = normalize_clip(coco_2d)     # (T, 17, 2)

        T = normalized.shape[0]
        for start in range(0, T - seq_len + 1, step):
            clip = normalized[start : start + seq_len]
            if np.abs(clip).max() < 5.0:  # reject normalization failures
                all_clips.append(clip)

    if not all_clips:
        return np.empty((0, seq_len, N_JOINTS, 2), dtype=np.float32)

    return np.stack(all_clips).astype(np.float32)
