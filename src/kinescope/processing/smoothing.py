"""
Interpolation and smoothing for pose timeseries.

Ported from utils/processing.py (Chambers et al. 2020 pipeline).
"""

import pandas as pd


def interpolate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate missing x, y values (sorted by frame)."""
    df = df.sort_values("frame")
    df["x"] = df["x"].interpolate()
    df["y"] = df["y"].interpolate()
    return df


def smooth(
    df: pd.DataFrame,
    var: str,
    winmed: float,
    winmean: float,
) -> pd.DataFrame:
    """
    Apply median-then-mean rolling smoothing to column `var`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'frame', 'fps', and `var` columns.
    var : str
        Column to smooth (in-place).
    winmed : float
        Median window in seconds (0 = skip).
    winmean : float
        Mean window in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with `var` smoothed and a 'smooth' column recording winmed.
    """
    fps = df["fps"].unique()[0]
    winmed_frames = int(winmed * fps)
    winmean_frames = int(winmean * fps)
    df = df.reset_index(drop=True)

    if winmed_frames > 0:
        x = df.sort_values("frame")[var].rolling(center=True, window=winmed_frames).median()
        df[var] = x.rolling(center=True, window=winmean_frames).mean()

    df["smooth"] = winmed
    return df
