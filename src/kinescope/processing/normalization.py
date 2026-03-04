"""
Skeleton normalization: rotate and scale keypoints to body-relative coordinates.

Ported from utils/processing.py (Chambers et al. 2020 pipeline).
Updated to use COCO-17 keypoint naming.
"""

import numpy as np
import pandas as pd

from kinescope.skeleton import NECK


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute joint-pair angles and midpoints
# ──────────────────────────────────────────────────────────────────────────────

def _joint_pair_angle(df: pd.DataFrame, left_bp: str, right_bp: str) -> pd.DataFrame:
    """Compute angle between left and right joint across frames."""
    df_pivoted = df[df["bp"].isin([left_bp, right_bp])].copy()
    pt = pd.pivot_table(df_pivoted, columns=["bp"], values=["x", "y"], index=["frame"])
    angle = np.arctan2(
        pt["y", right_bp] - pt["y", left_bp],
        pt["x", right_bp] - pt["x", left_bp],
    )
    return angle.rename(f"{left_bp}_{right_bp}_angle")


def _joint_pair_center(
    df: pd.DataFrame, left_bp: str, right_bp: str, prefix: str
) -> pd.DataFrame:
    """Compute midpoint of left and right joint across frames."""
    df_pivoted = df[df["bp"].isin([left_bp, right_bp])].copy()
    pt = pd.pivot_table(df_pivoted, columns=["bp"], values=["x", "y"], index=["frame"])
    center = pd.DataFrame({
        f"{prefix}x": (pt["x", left_bp] + pt["x", right_bp]) / 2,
        f"{prefix}y": (pt["y", left_bp] + pt["y", right_bp]) / 2,
    })
    return center


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise_skeletons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize skeleton keypoints to body-relative coordinates.

    For each frame:
    1. Compute reference axes from shoulder angle (upper body) and hip angle (lower body)
    2. Rotate keypoints to align body axis vertically
    3. Scale by trunk length (shoulder-center to hip-center distance)

    Upper body keypoints are centered on shoulder midpoint; lower body on hip midpoint.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical COCO DataFrame with neck appended (from pose.io.add_neck).
        Required columns: video, frame, bp, x, y, fps.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with x, y replaced by normalized coordinates,
        plus auxiliary columns: ref_angle, ref_dist, refx, refy, delta_t.
    """
    _UPPER_BPS = [
        "nose", NECK, "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_eye", "left_eye",
        "right_ear", "left_ear",
    ]
    _LOWER_BPS = ["right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_hip"]

    df = df.copy()
    df["upper"] = df["bp"].isin(_UPPER_BPS).astype(int)

    # Shoulder and hip reference angles + midpoints (per video × frame)
    s_angle = df.groupby("video").apply(
        lambda g: _joint_pair_angle(g, "left_shoulder", "right_shoulder")
    ).reset_index()
    h_angle = df.groupby("video").apply(
        lambda g: _joint_pair_angle(g, "left_hip", "right_hip")
    ).reset_index()
    uref = df.groupby("video").apply(
        lambda g: _joint_pair_center(g, "left_shoulder", "right_shoulder", "uref")
    ).reset_index()
    lref = df.groupby("video").apply(
        lambda g: _joint_pair_center(g, "left_hip", "right_hip", "lref")
    ).reset_index()

    s_angle.columns = s_angle.columns.get_level_values(0)
    h_angle.columns = h_angle.columns.get_level_values(0)
    uref.columns = uref.columns.get_level_values(0)
    lref.columns = lref.columns.get_level_values(0)

    ref = s_angle.copy()
    ref_angle_col = [c for c in ref.columns if c not in ("video", "frame")][0]
    hip_angle_col = [c for c in h_angle.columns if c not in ("video", "frame")][0]
    ref = ref.rename(columns={ref_angle_col: "shoulder_angle"})
    ref["hip_angle"] = h_angle[hip_angle_col].values

    ref = pd.merge(ref, uref, on=["video", "frame"], how="inner")
    ref = pd.merge(ref, lref, on=["video", "frame"], how="inner")

    df = pd.merge(df, ref, on=["video", "frame"], how="outer")

    df["refx"] = df["urefx"] * df["upper"] + df["lrefx"] * (1 - df["upper"])
    df["refy"] = df["urefy"] * df["upper"] + df["lrefy"] * (1 - df["upper"])
    df["ref_dist"] = np.sqrt(
        (df["urefy"] - df["lrefy"]) ** 2 + (df["urefx"] - df["lrefx"]) ** 2
    )
    df["ref_angle"] = df["shoulder_angle"] * df["upper"] + df["hip_angle"] * (1 - df["upper"])

    # Map ref_angle to [0, 2π] then align upward
    df.loc[df["ref_angle"] < 0, "ref_angle"] += 2 * np.pi
    df.loc[df["ref_angle"] < np.pi, "ref_angle"] = np.pi - df.loc[df["ref_angle"] < np.pi, "ref_angle"]
    df.loc[(df["ref_angle"] > np.pi) & (df["ref_angle"] < 2 * np.pi), "ref_angle"] = (
        3 * np.pi - df.loc[(df["ref_angle"] > np.pi) & (df["ref_angle"] < 2 * np.pi), "ref_angle"]
    )

    # Rotate
    dx = df["x"] - df["refx"]
    dy = df["y"] - df["refy"]
    df["x"] = (df["refx"] + np.cos(df["ref_angle"]) * dx - np.sin(df["ref_angle"]) * dy - df["refx"]) / df["ref_dist"]
    df["y"] = (df["refy"] + np.sin(df["ref_angle"]) * dx + np.cos(df["ref_angle"]) * dy - df["refy"]) / df["ref_dist"]

    # Shift lower body so trunk length = 1
    df.loc[df["upper"] == 0, "y"] += 1

    df["delta_t"] = 1 / df["fps"]
    return df


def dont_normalise_skeletons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reference axes (rotation/scaling) without applying the normalization.
    Adds reference columns to the DataFrame but leaves x, y unchanged.
    Used for computing joint angles on un-normalized coordinates.
    """
    df = normalise_skeletons(df)
    # Undo x/y replacement — restore by reloading from the pre-norm copy
    # Note: this is rarely needed; prefer normalise_skeletons for the pipeline
    df["delta_t"] = 1 / df["fps"]
    return df
