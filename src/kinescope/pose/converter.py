"""
Convert pose data from common formats to COCO-17 canonical DataFrame.

Supported input formats:
  - "coco"       : COCO-17 CSV (sam3d, MMPose, etc.) — passthrough
  - "openpose"   : OpenPose JSON (18-keypoint body_18 format)
  - "mediapipe"  : MediaPipe Pose CSV/DataFrame (33 BlazePose landmarks)
  - "kinect_v2"  : KinectV2 CSV/DataFrame (25 joints)
  - "smpl"       : SMPL joint positions CSV/DataFrame (24 joints)
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from kinescope.skeleton import COCO_PART_NAMES
from kinescope.pose.io import read_coco_csv

SUPPORTED_FORMATS = ["coco", "openpose", "mediapipe", "kinect_v2", "smpl"]

# ──────────────────────────────────────────────────────────────────────────────
# Index maps: source_index → coco_index (None = no equivalent)
# ──────────────────────────────────────────────────────────────────────────────

# OpenPose-18 → COCO-17  (Neck at index 1 has no COCO equivalent)
_OPENPOSE_TO_COCO = {
    0: 0,   # Nose → nose
    1: None,# Neck → (dropped)
    2: 6,   # RShoulder → right_shoulder
    3: 8,   # RElbow → right_elbow
    4: 10,  # RWrist → right_wrist
    5: 5,   # LShoulder → left_shoulder
    6: 7,   # LElbow → left_elbow
    7: 9,   # LWrist → left_wrist
    8: 12,  # RHip → right_hip
    9: 14,  # RKnee → right_knee
    10: 16, # RAnkle → right_ankle
    11: 11, # LHip → left_hip
    12: 13, # LKnee → left_knee
    13: 15, # LAnkle → left_ankle
    14: 2,  # REye → right_eye
    15: 1,  # LEye → left_eye
    16: 4,  # REar → right_ear
    17: 3,  # LEar → left_ear
}

# MediaPipe-33 (BlazePose) → COCO-17
_MEDIAPIPE_TO_COCO = {
    0: 0,   # nose → nose
    2: 1,   # left_eye → left_eye
    5: 2,   # right_eye → right_eye
    7: 3,   # left_ear → left_ear
    8: 4,   # right_ear → right_ear
    11: 5,  # left_shoulder → left_shoulder
    12: 6,  # right_shoulder → right_shoulder
    13: 7,  # left_elbow → left_elbow
    14: 8,  # right_elbow → right_elbow
    15: 9,  # left_wrist → left_wrist
    16: 10, # right_wrist → right_wrist
    23: 11, # left_hip → left_hip
    24: 12, # right_hip → right_hip
    25: 13, # left_knee → left_knee
    26: 14, # right_knee → right_knee
    27: 15, # left_ankle → left_ankle
    28: 16, # right_ankle → right_ankle
}

# KinectV2-25 → COCO-17  (spine, head, hand tips → None)
_KINECT_V2_TO_COCO = {
    0: None,  # SpineBase
    1: None,  # SpineMid
    2: None,  # Neck
    3: None,  # Head
    4: 5,     # ShoulderLeft → left_shoulder
    5: 7,     # ElbowLeft → left_elbow
    6: 9,     # WristLeft → left_wrist
    7: None,  # HandLeft (duplicate of WristLeft)
    8: 6,     # ShoulderRight → right_shoulder
    9: 8,     # ElbowRight → right_elbow
    10: 10,   # WristRight → right_wrist
    11: None, # HandRight (duplicate of WristRight)
    12: 11,   # HipLeft → left_hip
    13: 13,   # KneeLeft → left_knee
    14: 15,   # AnkleLeft → left_ankle
    15: None, # FootLeft (duplicate of AnkleLeft)
    16: 12,   # HipRight → right_hip
    17: 14,   # KneeRight → right_knee
    18: 16,   # AnkleRight → right_ankle
    19: None, # FootRight (duplicate of AnkleRight)
    20: None, # SpineShoulder
    21: None, # HandTipLeft
    22: None, # ThumbLeft
    23: None, # HandTipRight
    24: None, # ThumbRight
}

# SMPL-24 → COCO-17  (spine, collar, head, feet → None)
_SMPL_TO_COCO = {
    0: None,  # Pelvis
    1: 11,    # LeftHip → left_hip
    2: 12,    # RightHip → right_hip
    3: None,  # Spine1
    4: 13,    # LeftKnee → left_knee
    5: 14,    # RightKnee → right_knee
    6: None,  # Spine2
    7: 15,    # LeftAnkle → left_ankle
    8: 16,    # RightAnkle → right_ankle
    9: None,  # Spine3
    10: None, # LeftFoot
    11: None, # RightFoot
    12: None, # Neck
    13: None, # LeftCollar
    14: None, # RightCollar
    15: None, # Head
    16: 5,    # LeftShoulder → left_shoulder
    17: 6,    # RightShoulder → right_shoulder
    18: 7,    # LeftElbow → left_elbow
    19: 8,    # RightElbow → right_elbow
    20: 9,    # LeftWrist → left_wrist
    21: 10,   # RightWrist → right_wrist
    22: None, # LeftHand (duplicate of wrist)
    23: None, # RightHand (duplicate of wrist)
}


# ──────────────────────────────────────────────────────────────────────────────
# Private converters
# ──────────────────────────────────────────────────────────────────────────────

def _apply_index_map(
    df_src: pd.DataFrame,
    index_map: dict,
    frame_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = None,
    conf_col: str = None,
    video_name: str = None,
    fps: float = None,
) -> pd.DataFrame:
    """
    Generic helper: given a wide-per-frame or long-per-joint DataFrame and an
    index_map {src_idx: coco_idx}, produce the canonical COCO-17 DataFrame.
    Unmapped or missing joints get x=y=NaN, confidence=0.0.
    """
    has_z = z_col is not None and z_col in df_src.columns
    rows = []
    frames = df_src[frame_col].unique()

    for frame in frames:
        frame_data = df_src[df_src[frame_col] == frame]
        # Build full COCO-17 output (17 rows per frame)
        coco_row = {i: {"x": np.nan, "y": np.nan, "confidence": 0.0} for i in range(17)}
        if has_z:
            for r in coco_row.values():
                r["z"] = np.nan

        for _, src_row in frame_data.iterrows():
            src_idx = int(src_row.get("part_idx", src_row.name))
            coco_idx = index_map.get(src_idx)
            if coco_idx is None:
                continue
            coco_row[coco_idx]["x"] = src_row[x_col]
            coco_row[coco_idx]["y"] = src_row[y_col]
            if has_z:
                coco_row[coco_idx]["z"] = src_row[z_col]
            coco_row[coco_idx]["confidence"] = float(src_row[conf_col]) if conf_col and conf_col in src_row else 1.0

        for coco_idx, vals in coco_row.items():
            row = {
                "frame": frame,
                "part_idx": coco_idx,
                "bp": COCO_PART_NAMES[coco_idx],
                **vals,
            }
            if video_name:
                row["video"] = video_name
            if fps is not None:
                row["fps"] = fps
            rows.append(row)

    cols = ["frame", "bp", "part_idx", "x", "y"]
    if has_z:
        cols.append("z")
    cols.append("confidence")
    if video_name:
        cols = ["video"] + cols
    if fps is not None:
        cols.append("fps")

    return pd.DataFrame(rows)[cols].reset_index(drop=True)


def _from_coco(source, video_name=None, fps=None) -> pd.DataFrame:
    """Passthrough: source is already COCO-17 CSV."""
    return read_coco_csv(source, video_name=video_name, fps=fps)


def _from_openpose(source, video_name=None, fps=None) -> pd.DataFrame:
    """Convert OpenPose JSON (body_18) to COCO-17."""
    source = Path(source)
    with open(source) as f:
        data = json.load(f)

    rows = []
    # OpenPose JSON: list of frames or dict with 'people' per frame
    if isinstance(data, list):
        frames_data = data
    else:
        frames_data = [data]

    for frame_entry in frames_data:
        frame_id = frame_entry.get("frame_id", frame_entry.get("frame", 0))
        people = frame_entry.get("people", frame_entry.get("instances", []))
        if not people:
            continue
        # Pick person with highest total confidence
        person = max(
            people,
            key=lambda p: sum(p.get("keypoint_scores", p.get("pose_keypoints_2d", [0])[2::3]))
        )
        kps = person.get("keypoints", person.get("pose_keypoints_2d", []))
        scores = person.get("keypoint_scores", [])

        # Flatten if kps is flat [x0,y0,c0, x1,y1,c1, ...]
        if kps and not isinstance(kps[0], (list, tuple)):
            kps_parsed = [(kps[i], kps[i+1]) for i in range(0, min(len(kps), 18*3), 3)]
            if not scores:
                scores = [kps[i+2] for i in range(0, min(len(kps), 18*3), 3)]
        else:
            kps_parsed = [tuple(k[:2]) for k in kps]

        for op_idx, (x, y) in enumerate(kps_parsed[:18]):
            coco_idx = _OPENPOSE_TO_COCO.get(op_idx)
            if coco_idx is None:
                continue
            conf = float(scores[op_idx]) if op_idx < len(scores) else 1.0
            row = {"frame": frame_id, "part_idx": coco_idx,
                   "bp": COCO_PART_NAMES[coco_idx], "x": x, "y": y, "confidence": conf}
            if video_name:
                row["video"] = video_name
            if fps is not None:
                row["fps"] = fps
            rows.append(row)

    return _fill_missing_keypoints(pd.DataFrame(rows), video_name, fps)


def _from_mediapipe(source, video_name=None, fps=None) -> pd.DataFrame:
    """Convert MediaPipe Pose (33 landmarks) CSV/DataFrame to COCO-17."""
    df = _load_source(source)
    # MediaPipe CSV expected columns: frame, landmark_index, x, y, [z], [visibility]
    if "landmark_index" not in df.columns and "part_idx" not in df.columns:
        raise ValueError("MediaPipe source must have 'landmark_index' or 'part_idx' column")
    idx_col = "landmark_index" if "landmark_index" in df.columns else "part_idx"
    df = df.rename(columns={idx_col: "part_idx"})
    conf_col = "visibility" if "visibility" in df.columns else None
    z_col = "z" if "z" in df.columns else None
    return _apply_index_map(df, _MEDIAPIPE_TO_COCO, conf_col=conf_col, z_col=z_col,
                            video_name=video_name, fps=fps)


def _from_kinect_v2(source, video_name=None, fps=None) -> pd.DataFrame:
    """Convert KinectV2 (25 joints) CSV/DataFrame to COCO-17."""
    df = _load_source(source)
    if "joint_index" not in df.columns and "part_idx" not in df.columns:
        raise ValueError("KinectV2 source must have 'joint_index' or 'part_idx' column")
    idx_col = "joint_index" if "joint_index" in df.columns else "part_idx"
    df = df.rename(columns={idx_col: "part_idx"})
    z_col = "z" if "z" in df.columns else None
    return _apply_index_map(df, _KINECT_V2_TO_COCO, z_col=z_col, video_name=video_name, fps=fps)


def _from_smpl(source, video_name=None, fps=None) -> pd.DataFrame:
    """Convert SMPL (24 joints) CSV/DataFrame to COCO-17."""
    df = _load_source(source)
    if "joint_index" not in df.columns and "part_idx" not in df.columns:
        raise ValueError("SMPL source must have 'joint_index' or 'part_idx' column")
    idx_col = "joint_index" if "joint_index" in df.columns else "part_idx"
    df = df.rename(columns={idx_col: "part_idx"})
    z_col = "z" if "z" in df.columns else None
    return _apply_index_map(df, _SMPL_TO_COCO, z_col=z_col, video_name=video_name, fps=fps)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_source(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source)


def _fill_missing_keypoints(df: pd.DataFrame, video_name, fps) -> pd.DataFrame:
    """Ensure all 17 COCO keypoints exist for every frame; fill missing with NaN/0."""
    if df.empty:
        return df
    frames = df["frame"].unique()
    existing = set(zip(df["frame"], df["part_idx"]))
    missing_rows = []
    for frame in frames:
        for coco_idx in range(17):
            if (frame, coco_idx) not in existing:
                row = {"frame": frame, "part_idx": coco_idx,
                       "bp": COCO_PART_NAMES[coco_idx],
                       "x": np.nan, "y": np.nan, "confidence": 0.0}
                if video_name:
                    row["video"] = video_name
                if fps is not None:
                    row["fps"] = fps
                missing_rows.append(row)
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)
    return df.sort_values(["frame", "part_idx"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_CONVERTERS = {
    "coco":       _from_coco,
    "openpose":   _from_openpose,
    "mediapipe":  _from_mediapipe,
    "kinect_v2":  _from_kinect_v2,
    "smpl":       _from_smpl,
}


def convert_to_coco(
    source: Union[str, Path, pd.DataFrame],
    fmt: str,
    video_name: str = None,
    fps: float = None,
) -> pd.DataFrame:
    """
    Convert pose data from a supported format to COCO-17 canonical DataFrame.

    Parameters
    ----------
    source : str, Path, or DataFrame
        Input data — file path for file-based formats, DataFrame for in-memory.
    fmt : str
        Source format. One of: "coco", "openpose", "mediapipe", "kinect_v2", "smpl".
    video_name : str, optional
        Video identifier embedded in 'video' column.
    fps : float, optional
        Frames per second embedded in 'fps' column.

    Returns
    -------
    pd.DataFrame
        Canonical COCO-17 DataFrame:
        frame, [video], bp, part_idx, x, y, [z], confidence, [fps]
        One row per keypoint per frame; missing keypoints get x=y=NaN, confidence=0.
    """
    fmt = fmt.lower()
    if fmt not in _CONVERTERS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {SUPPORTED_FORMATS}")
    return _CONVERTERS[fmt](source, video_name=video_name, fps=fps)
