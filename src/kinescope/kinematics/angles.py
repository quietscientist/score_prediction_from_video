"""
Joint angle computation from COCO-17 keypoint coordinates.

Ported from utils/kinematics.py (Chambers et al. 2020 pipeline).
Updated to use COCO-17 keypoint naming.
"""

import numpy as np
import pandas as pd

from kinescope.skeleton import JOINT_ANGLE_TRIPLETS, NECK


def get_joint_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute joint angles from x, y keypoint coordinates.

    Requires a 'neck' keypoint row (added by pose.io.add_neck before calling).
    Angles are computed via arctan2 of the cross and dot products of the
    two vectors meeting at each joint vertex.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical COCO DataFrame with neck appended.
        Required columns: video, frame, bp, x, y, fps, time.

    Returns
    -------
    pd.DataFrame
        Columns: video, frame, bp, side, part, angle, fps, time, delta_t
        One row per joint per frame.
        - Elbow/knee: absolute flexion angle (0–180°)
        - Shoulder/hip: angle in 0–360° space
    """
    df = df[~df["x"].isnull()].copy()
    df["delta_t"] = 1 / df["fps"]

    rows = []
    frames = df["frame"].unique()

    for frame in frames:
        fdf = df[df["frame"] == frame].set_index("bp")
        fps_val = fdf["fps"].iloc[0]
        time_val = fdf["time"].iloc[0]
        dt_val = 1 / fps_val

        for vertex, proximal, distal in JOINT_ANGLE_TRIPLETS:
            # All three keypoints must be present
            if not all(k in fdf.index for k in (vertex, proximal, distal)):
                continue

            x0, y0 = fdf.loc[vertex, "x"], fdf.loc[vertex, "y"]
            x1, y1 = fdf.loc[proximal, "x"], fdf.loc[proximal, "y"]
            x2, y2 = fdf.loc[distal, "x"], fdf.loc[distal, "y"]

            # Vectors from vertex to proximal and distal
            dot = (x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0)
            det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            angle_deg = np.arctan2(det, dot) * 180 / np.pi

            # side and part from vertex name (e.g. "left_elbow" → side="left", part="elbow")
            parts = vertex.split("_", 1)
            side = parts[0]  # "left" or "right"
            part = parts[1] if len(parts) > 1 else vertex

            # Flip left-side shoulder/hip so L and R angles are comparable
            if side == "left" and part in ("shoulder", "hip"):
                angle_deg = -angle_deg

            # Elbow/knee: use absolute flexion (0–180°)
            if part in ("elbow", "knee"):
                angle_deg = abs(angle_deg)

            # Shoulder/hip: map to 0–360°
            if part in ("shoulder", "hip") and angle_deg < 0:
                angle_deg += 360

            rows.append({
                "video": fdf["video"].iloc[0],
                "frame": frame,
                "bp": vertex,
                "side": side,
                "part": part,
                "angle": angle_deg,
                "fps": fps_val,
                "time": time_val,
                "delta_t": dt_val,
            })

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
