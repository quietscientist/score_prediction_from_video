"""
Statistical feature extraction from kinematic timeseries.

Ported from utils/kinematics.py (Chambers et al. 2020 pipeline).
"""

import numpy as np
import pandas as pd
import scipy.stats

from kinescope._vendor import circstat as CS


# ──────────────────────────────────────────────────────────────────────────────
# Entropy
# ──────────────────────────────────────────────────────────────────────────────

def ent(data: pd.Series) -> float:
    """Shannon entropy of a discrete distribution derived from value counts."""
    p = data.value_counts() / len(data)
    return scipy.stats.entropy(p)


# ──────────────────────────────────────────────────────────────────────────────
# Total (whole-video) features
# ──────────────────────────────────────────────────────────────────────────────

def xy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate XY kinematic features over the full video for one body part.

    Parameters
    ----------
    df : pd.DataFrame
        XY dynamics DataFrame for a single (bp, video) group.

    Returns
    -------
    pd.DataFrame with one row per video+bp containing:
        medianx, mediany, IQRx, IQRy,
        medianvelx, medianvely, IQRvelx, IQRvely,
        IQRaccx, IQRaccy, meanent
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame([{
        "video": df["video"].iloc[0],
        "bp": df["bp"].iloc[0],
        "medianx": df["x"].median(),
        "mediany": df["y"].median(),
        "IQRx": df["x"].quantile(0.75) - df["x"].quantile(0.25),
        "IQRy": df["y"].quantile(0.75) - df["y"].quantile(0.25),
        "medianvelx": np.abs(df["velocity_x"]).median(),
        "medianvely": np.abs(df["velocity_y"]).median(),
        "IQRvelx": df["velocity_x"].quantile(0.75) - df["velocity_x"].quantile(0.25),
        "IQRvely": df["velocity_y"].quantile(0.75) - df["velocity_y"].quantile(0.25),
        "IQRaccx": df["acceleration_x"].quantile(0.75) - df["acceleration_x"].quantile(0.25),
        "IQRaccy": df["acceleration_y"].quantile(0.75) - df["acceleration_y"].quantile(0.25),
        "meanent": (ent(df["x"].round(2)) + ent(df["y"].round(2))) / 2,
    }])


def angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate angular kinematic features over the full video for one joint.

    Parameters
    ----------
    df : pd.DataFrame
        Angle dynamics DataFrame for a single (bp, video) group.

    Returns
    -------
    pd.DataFrame with one row per video+bp containing:
        mean_angle, stdev_angle, entropy_angle,
        median_vel_angle, IQR_vel_angle, IQR_acc_angle
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    angles_rad = np.radians(df["angle"].dropna())
    return pd.DataFrame([{
        "video": df["video"].iloc[0],
        "bp": df["bp"].iloc[0],
        "mean_angle": np.degrees(CS.nanmean(np.array(angles_rad))),
        "stdev_angle": np.sqrt(np.degrees(CS.nanvar(np.array(angles_rad)))),
        "entropy_angle": ent(df["angle"].round()),
        "median_vel_angle": np.abs(df["velocity"]).median(),
        "IQR_vel_angle": df["velocity"].quantile(0.75) - df["velocity"].quantile(0.25),
        "IQR_acc_angle": df["acceleration"].quantile(0.75) - df["acceleration"].quantile(0.25),
    }])


def corr_lr(df: pd.DataFrame, var: str) -> float:
    """
    Pearson correlation between left and right sides for variable `var`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'side' column with values "left"/"right".
    var : str
        Column to correlate.
    """
    idf = pd.DataFrame({
        "R": df[df["side"] == "right"].reset_index()[var],
        "L": df[df["side"] == "left"].reset_index()[var],
    })
    return idf.corr().loc["L", "R"]


# ──────────────────────────────────────────────────────────────────────────────
# Rolling (windowed) features
# ──────────────────────────────────────────────────────────────────────────────

def rolling_xy_features(
    df: pd.DataFrame,
    window_size: int = 30,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling window XY features for one (bp, video) group.

    Returns one row per frame with windowed statistics.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    video = df["video"].iloc[0]
    bp = df["bp"].iloc[0]

    out = pd.DataFrame({"frame": df["frame"], "video": video, "bp": bp})
    w = df.rolling(window=window_size, min_periods=min_periods)

    out["median_x"]     = w["x"].median()
    out["median_y"]     = w["y"].median()
    out["IQR_x"]        = w["x"].quantile(0.75) - w["x"].quantile(0.25)
    out["IQR_y"]        = w["y"].quantile(0.75) - w["y"].quantile(0.25)
    out["median_vel_x"] = w["velocity_x"].apply(lambda s: np.abs(s).median())
    out["median_vel_y"] = w["velocity_y"].apply(lambda s: np.abs(s).median())
    out["IQR_vel_x"]    = w["velocity_x"].quantile(0.75) - w["velocity_x"].quantile(0.25)
    out["IQR_vel_y"]    = w["velocity_y"].quantile(0.75) - w["velocity_y"].quantile(0.25)
    out["IQR_acc_x"]    = w["acceleration_x"].quantile(0.75) - w["acceleration_x"].quantile(0.25)
    out["IQR_acc_y"]    = w["acceleration_y"].quantile(0.75) - w["acceleration_y"].quantile(0.25)
    out["mean_ent"]     = w["x"].apply(lambda s: ent(s.round(2)))

    return out


def rolling_angle_features(
    df: pd.DataFrame,
    window_size: int = 30,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling window angle features for one (bp, video) group.

    Returns one row per frame with windowed statistics.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    video = df["video"].iloc[0]
    bp = df["bp"].iloc[0]

    out = pd.DataFrame({"frame": df["frame"], "video": video, "bp": bp})
    w = df.rolling(window=window_size, min_periods=min_periods)

    out["mean_angle"]   = w["angle"].mean()
    out["stdev_angle"]  = w["angle"].std()
    out["entropy_angle"] = w["angle"].apply(lambda s: ent(s.round()))
    out["median_vel"]   = w["velocity"].apply(lambda s: np.abs(s).median())
    out["IQR_vel"]      = w["velocity"].quantile(0.75) - w["velocity"].quantile(0.25)
    out["IQR_acc"]      = w["acceleration"].quantile(0.75) - w["acceleration"].quantile(0.25)

    return out


def rolling_corr_lr(
    df: pd.DataFrame,
    window_size: int = 30,
    min_periods: int = 1,
    var: str = "angle",
) -> pd.DataFrame:
    """
    Rolling left-right correlation for variable `var`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'side' column ("left"/"right") and `var` column.

    Returns
    -------
    pd.DataFrame with columns: frame, rolling_corr
    """
    right = df[df["side"] == "right"].set_index("frame")[var].rename("R")
    left  = df[df["side"] == "left"].set_index("frame")[var].rename("L")
    combined = pd.concat([right, left], axis=1).sort_index()
    combined["rolling_corr"] = (
        combined["R"].rolling(window=window_size, min_periods=min_periods)
        .corr(combined["L"].rolling(window=window_size, min_periods=min_periods))
    )
    return combined.reset_index()[["frame", "rolling_corr"]]
