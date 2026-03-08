"""
NTU RGB+D 120 dataset loader: .skeleton files → normalized COCO-17 clips.

Directory layout expected:
    ntu120/
    └── nturgbd_skeletons_s001_to_s017/   (or any subdirectory structure)
        ├── S001C001P001R001A001.skeleton
        ├── S001C001P001R001A002.skeleton
        └── ...

Each .skeleton file format (NTU RGB+D):
    Line 1:  number of frames
    Per frame:
        number of bodies
        body_id tracking_state ...
        number of joints (25)
        per joint: x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ trackingState

Only the first body in each frame is used.

NTU-25 → COCO-17 joint mapping
---------------------------------
Joints are ordered by LIMB in the NTU file (all left-side joints consecutive, then right).
This matches the Kinect SDK 0-based order + 1 offset.
Source: PointNames.txt in the rekinect-pose repo.

NTU joint index (1-based) → COCO-17 index:
    4  (Head)              → 0  nose  (approximate; NTU joint 4 per PointNames.txt)
    5  (ShoulderLeft)      → 5  left_shoulder
    6  (ElbowLeft)         → 7  left_elbow
    7  (WristLeft)         → 9  left_wrist
    9  (ShoulderRight)     → 6  right_shoulder
    10 (ElbowRight)        → 8  right_elbow
    11 (WristRight)        → 10 right_wrist
    13 (HipLeft)           → 11 left_hip
    14 (KneeLeft)          → 13 left_knee
    15 (AnkleLeft)         → 15 left_ankle
    17 (HipRight)          → 12 right_hip
    18 (KneeRight)         → 14 right_knee
    19 (AnkleRight)        → 16 right_ankle
    -- not present         → 1-4 (eyes, ears): set to 0

References:
    NTU RGB+D 120: Liu et al. 2020, TPAMI
    Joint naming: https://github.com/shahroudy/NTURGB-D
"""

import pathlib

import numpy as np

from kinescope.pretrain._normalize_clip import normalize_clip
from kinescope.skeleton import COCO_PART_NAMES

N_JOINTS = len(COCO_PART_NAMES)  # 17

# NTU joint index (1-based) → COCO-17 index
# Joints are ordered by limb: left side (indices 5-8, 13-16), then right (9-12, 17-20).
_NTU_TO_COCO = {
    4:  0,   # Head          → nose (NTU 1-based joint 4 = Head per PointNames.txt)
    5:  5,   # ShoulderLeft  → left_shoulder
    6:  7,   # ElbowLeft     → left_elbow
    7:  9,   # WristLeft     → left_wrist
    9:  6,   # ShoulderRight → right_shoulder
    10: 8,   # ElbowRight    → right_elbow
    11: 10,  # WristRight    → right_wrist
    13: 11,  # HipLeft       → left_hip
    14: 13,  # KneeLeft      → left_knee
    15: 15,  # AnkleLeft     → left_ankle
    17: 12,  # HipRight      → right_hip
    18: 14,  # KneeRight     → right_knee
    19: 16,  # AnkleRight    → right_ankle
}


def _parse_skeleton_file(path: pathlib.Path) -> np.ndarray:
    """
    Parse a NTU RGB+D .skeleton file.

    Returns (T, 17, 2) float32 array in COCO-17 order (X, Y pixel coords).
    Returns None if the file is malformed or has zero valid frames.
    """
    try:
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception:
        return None

    idx = 0
    try:
        n_frames = int(lines[idx]); idx += 1
    except (IndexError, ValueError):
        return None

    sequence = np.zeros((n_frames, N_JOINTS, 2), dtype=np.float32)

    for t in range(n_frames):
        try:
            n_bodies = int(lines[idx]); idx += 1
        except (IndexError, ValueError):
            break

        body_read = False
        for b in range(n_bodies):
            try:
                # Body header: bodyID, clipedEdges, handLeftConfidence, handLeftState,
                #              handRightConfidence, handRightState, isResticted,
                #              leanX, leanY, trackingState
                idx += 1  # skip body header line
                n_joints = int(lines[idx]); idx += 1
            except (IndexError, ValueError):
                break

            for j in range(n_joints):
                try:
                    vals = lines[idx].split(); idx += 1
                except IndexError:
                    break

                if not body_read and b == 0:
                    # NTU joint index is 1-based
                    ntu_j = j + 1
                    coco_j = _NTU_TO_COCO.get(ntu_j)
                    if coco_j is not None:
                        # vals: x y z depthX depthY colorX colorY orientW orientX orientY orientZ trackState
                        # Use colorX, colorY (image-space) if available, else world X, Y
                        if len(vals) >= 7:
                            sequence[t, coco_j, 0] = float(vals[5])  # colorX
                            sequence[t, coco_j, 1] = float(vals[6])  # colorY
                        elif len(vals) >= 2:
                            sequence[t, coco_j, 0] = float(vals[0])  # world X
                            sequence[t, coco_j, 1] = float(vals[1])  # world Y

            if b == 0:
                body_read = True

    if n_frames == 0:
        return None

    # Eyes and ears (1-4) are not in NTU; copy nose so normalize_clip sees valid coords
    sequence[:, 1:5] = sequence[:, 0:1]

    return sequence


def load_ntu_clips(
    data_dir: pathlib.Path,
    seq_len: int = 60,
    max_files: int = None,
) -> np.ndarray:
    """
    Load NTU RGB+D 120 .skeleton files, convert to COCO-17, normalize, and chunk.

    Parameters
    ----------
    data_dir : Path — NTU directory containing .skeleton files (searched recursively)
    seq_len : int — clip length in frames
    max_files : int or None — limit number of files

    Returns
    -------
    (N, seq_len, 17, 2) float32 array
    """
    data_dir = pathlib.Path(data_dir)
    skeleton_files = sorted(data_dir.rglob("*.skeleton"))

    if not skeleton_files:
        print(f"NTU: no .skeleton files in {data_dir}, skipping.")
        return np.empty((0, seq_len, N_JOINTS, 2), dtype=np.float32)

    if max_files is not None:
        skeleton_files = skeleton_files[:max_files]

    all_clips = []
    step = seq_len // 2

    for skel_path in skeleton_files:
        sequence = _parse_skeleton_file(skel_path)
        if sequence is None:
            continue

        T = sequence.shape[0]
        if T < seq_len:
            continue

        normalized = normalize_clip(sequence)  # (T, 17, 2)

        for start in range(0, T - seq_len + 1, step):
            all_clips.append(normalized[start : start + seq_len])

    if not all_clips:
        return np.empty((0, seq_len, N_JOINTS, 2), dtype=np.float32)

    return np.stack(all_clips).astype(np.float32)
