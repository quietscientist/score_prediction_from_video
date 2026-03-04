"""
Kinematic dynamics: velocity, acceleration, angular displacement.

Ported from utils/kinematics.py (Chambers et al. 2020 pipeline).
"""

import numpy as np
import pandas as pd


def get_delta(df: pd.DataFrame, inp: str, outp: str) -> pd.DataFrame:
    """Frame-to-frame difference of column `inp` → stored in `outp`."""
    x = df[inp]
    df[outp] = np.concatenate(([0], np.diff(x))) * (x * 0 + 1)
    return df


def smooth_dyn(df: pd.DataFrame, inp: str, outp: str, win: float) -> pd.DataFrame:
    """Rolling mean smoothing over a window of `win` seconds (uses fps column)."""
    fps = df["fps"].unique()[0]
    win_frames = int(win * fps)
    x = df[inp].interpolate()
    df[outp] = x.rolling(window=win_frames, center=False).mean()
    return df


def angular_disp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Smallest signed angular displacement from x to y (handles 360° wrap).
    Returns array of displacements in degrees.
    """
    candidates = np.asarray([y - x, y - x + 360, y - x - 360])
    idx = np.abs(candidates).argmin(axis=0)
    return np.asarray([candidates[idx[i], i] for i in range(len(x))])


def get_angle_displacement(df: pd.DataFrame, inp: str, outp: str) -> pd.DataFrame:
    """
    Frame-to-frame angular displacement using shortest circular path.
    Stores result in column `outp`.
    """
    df = df.sort_values("frame")
    angle = np.array(df[inp])
    disp = angular_disp(angle[:-1], angle[1:])
    df[outp] = np.concatenate(([0], disp))
    return df


def get_dynamics_xy(xdf: pd.DataFrame, delta_window: float) -> pd.DataFrame:
    """
    Compute XY linear kinematics: velocity, acceleration, speed.

    Parameters
    ----------
    xdf : pd.DataFrame
        Canonical COCO DataFrame with x, y, bp, video, fps, delta_t columns.
    delta_window : float
        Smoothing window in seconds for velocity and acceleration.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        d_x, d_y, displacement, velocity_x, velocity_y,
        delta_velocity_x, delta_velocity_y, acceleration_x, acceleration_y,
        acceleration_x2, acceleration_y2, speed_raw, speed, part, side
    """
    xdf = xdf[["video", "frame", "x", "y", "bp", "fps", "pixel_x", "pixel_y",
               "time", "delta_t", "part_idx"]].copy()

    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: get_delta(g, "x", "d_x")
    ).reset_index(drop=True)
    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: get_delta(g, "y", "d_y")
    ).reset_index(drop=True)

    xdf["displacement"] = np.sqrt(xdf["d_x"] ** 2 + xdf["d_y"] ** 2)
    xdf["velocity_x_raw"] = xdf["d_x"] / xdf["delta_t"]
    xdf["velocity_y_raw"] = xdf["d_y"] / xdf["delta_t"]

    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: smooth_dyn(g, "velocity_x_raw", "velocity_x", delta_window)
    ).reset_index(drop=True)
    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: smooth_dyn(g, "velocity_y_raw", "velocity_y", delta_window)
    ).reset_index(drop=True)

    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: get_delta(g, "velocity_x", "delta_velocity_x")
    ).reset_index(drop=True)
    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: get_delta(g, "velocity_y", "delta_velocity_y")
    ).reset_index(drop=True)

    xdf["acceleration_x_raw"] = xdf["delta_velocity_x"] / xdf["delta_t"]
    xdf["acceleration_y_raw"] = xdf["delta_velocity_y"] / xdf["delta_t"]

    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: smooth_dyn(g, "acceleration_x_raw", "acceleration_x", delta_window)
    ).reset_index(drop=True)
    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: smooth_dyn(g, "acceleration_y_raw", "acceleration_y", delta_window)
    ).reset_index(drop=True)

    xdf["acceleration_x2"] = xdf["acceleration_x"] ** 2
    xdf["acceleration_y2"] = xdf["acceleration_y"] ** 2
    xdf["speed_raw"] = xdf["displacement"] / xdf["delta_t"]
    xdf = xdf.groupby(["bp", "video"]).apply(
        lambda g: smooth_dyn(g, "speed_raw", "speed", delta_window)
    ).reset_index(drop=True)

    # side: "left" or "right" prefix
    xdf["side"] = xdf["bp"].str.split("_").str[0]
    xdf["part"] = xdf["bp"].str.split("_", n=1).str[1]

    return xdf


def get_dynamics_angle(adf: pd.DataFrame, delta_window: float) -> pd.DataFrame:
    """
    Compute angular kinematics: angular velocity and acceleration.

    Parameters
    ----------
    adf : pd.DataFrame
        DataFrame from get_joint_angles with 'angle', 'delta_t', 'fps' columns.
    delta_window : float
        Smoothing window in seconds.

    Returns
    -------
    pd.DataFrame
        Input with added columns:
        displacement (angular), velocity_raw, velocity,
        delta_velocity, acceleration_raw, acceleration, acceleration2, part, side
    """
    adf = adf.groupby(["bp", "video"]).apply(
        lambda g: get_angle_displacement(g, "angle", "displacement")
    ).reset_index(drop=True)

    adf["velocity_raw"] = adf["displacement"] / adf["delta_t"]

    adf = adf.groupby(["bp", "video"]).apply(
        lambda g: smooth_dyn(g, "velocity_raw", "velocity", delta_window)
    ).reset_index(drop=True)
    adf = adf.groupby(["bp", "video"]).apply(
        lambda g: get_delta(g, "velocity", "delta_velocity")
    ).reset_index(drop=True)

    adf["acceleration_raw"] = adf["delta_velocity"] / adf["delta_t"]

    adf = adf.groupby(["bp", "video"]).apply(
        lambda g: smooth_dyn(g, "acceleration_raw", "acceleration", delta_window)
    ).reset_index(drop=True)

    adf["acceleration2"] = adf["acceleration"] ** 2
    adf["side"] = adf["bp"].str.split("_").str[0]
    adf["part"] = adf["bp"].str.split("_", n=1).str[1]

    return adf
