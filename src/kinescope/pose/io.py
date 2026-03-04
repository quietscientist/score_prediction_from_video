"""
Read COCO-17 pose CSVs into the canonical kinescope DataFrame format.

Compatible with sam3d-video-pose output and any other tool that produces
COCO-17 keypoints in CSV form.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from kinescope.skeleton import COCO_PART_NAMES, NECK


def read_coco_csv(
    path: Union[str, Path],
    video_name: str = None,
    fps: float = None,
) -> pd.DataFrame:
    """
    Read a COCO-17 pose CSV into the canonical kinescope DataFrame.

    Accepted column sets (sam3d and compatible tools):
      - frame, x, y, part_idx
      - frame, x, y, z, part_idx
      - frame, x, y, part_idx, confidence
      - frame, x, y, z, part_idx, confidence

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    video_name : str, optional
        Video identifier to embed in the 'video' column.
        Defaults to the file stem.
    fps : float, optional
        Frames per second. Embedded in 'fps' column if provided.

    Returns
    -------
    pd.DataFrame
        Columns: frame, x, y, [z], bp (COCO name str), confidence, [fps]
        One row per keypoint per frame. 17 rows per frame.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"frame", "x", "y", "part_idx"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # Map part_idx → COCO keypoint name
    df["bp"] = df["part_idx"].map(lambda i: COCO_PART_NAMES[int(i)])

    # Confidence: default to 1.0 if not present
    if "confidence" not in df.columns:
        df["confidence"] = 1.0

    # Video identifier
    if video_name is None:
        video_name = path.stem
    df["video"] = video_name

    # FPS column
    if fps is not None:
        df["fps"] = fps

    # Select output columns
    cols = ["video", "frame", "bp", "part_idx", "x", "y"]
    if "z" in df.columns:
        cols.append("z")
    cols.append("confidence")
    if "fps" in df.columns:
        cols.append("fps")

    return df[cols].reset_index(drop=True)


def add_neck(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a synthetic 'neck' keypoint as the midpoint of left_shoulder and
    right_shoulder. Appends neck rows to the DataFrame.

    Used internally by normalise_skeletons; the neck is not part of COCO-17.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical COCO DataFrame (output of read_coco_csv or convert_to_coco).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with neck rows appended. One neck row per frame.
    """
    ls = df[df["bp"] == "left_shoulder"].set_index("frame")
    rs = df[df["bp"] == "right_shoulder"].set_index("frame")

    shared_frames = ls.index.intersection(rs.index)
    if len(shared_frames) == 0:
        return df

    neck_rows = []
    for frame in shared_frames:
        row = {"frame": frame, "bp": NECK, "part_idx": -1}
        row["x"] = (ls.loc[frame, "x"] + rs.loc[frame, "x"]) / 2
        row["y"] = (ls.loc[frame, "y"] + rs.loc[frame, "y"]) / 2
        row["confidence"] = min(ls.loc[frame, "confidence"], rs.loc[frame, "confidence"])
        if "z" in df.columns:
            row["z"] = (ls.loc[frame, "z"] + rs.loc[frame, "z"]) / 2
        if "video" in df.columns:
            row["video"] = ls.loc[frame, "video"]
        if "fps" in df.columns:
            row["fps"] = ls.loc[frame, "fps"]
        neck_rows.append(row)

    neck_df = pd.DataFrame(neck_rows)
    return pd.concat([df, neck_df], ignore_index=True)
