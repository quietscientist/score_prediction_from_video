"""
Own COCO-17 CSV loader: reads pose CSVs from the kinescope pipeline → normalized clips.

Accepts:
  - Directories of COCO-17 CSVs produced by sam3d-video-pose or kinescope process
  - Pre-processed .npy clip arrays of shape (N, T, 17, 2)

Directory layout expected:
    coco/
    └── *.csv     (COCO-17 CSV files, one per video)
"""

import pathlib

import numpy as np

from kinescope.skeleton import COCO_PART_NAMES
from kinescope.pretrain._normalize_clip import normalize_clip

N_JOINTS = len(COCO_PART_NAMES)  # 17


def load_coco_clips(
    data_dir: pathlib.Path,
    seq_len: int = 60,
) -> np.ndarray:
    """
    Load COCO-17 CSV files or pre-processed .npy files, normalize, and chunk into clips.

    Parameters
    ----------
    data_dir : Path — directory of .csv or .npy files
    seq_len : int — clip length in frames

    Returns
    -------
    (N, seq_len, 17, 2) float32 array
    """
    data_dir = pathlib.Path(data_dir)

    # .npy files: already clipped arrays, just concatenate
    npy_files = sorted(data_dir.glob("*.npy"))
    if npy_files:
        clips = np.concatenate([np.load(f) for f in npy_files], axis=0)
        return clips.astype(np.float32)

    # .csv files: load, normalize, chunk
    from kinescope.pose.io import read_coco_csv
    from kinescope.processing.smoothing import interpolate_df

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No .csv or .npy files found in {data_dir}."
        )

    part_index = {name: i for i, name in enumerate(COCO_PART_NAMES)}
    all_clips = []
    step = seq_len // 2

    for csv_path in csv_files:
        try:
            df = read_coco_csv(csv_path)
            df = interpolate_df(df)
        except Exception:
            continue

        frames = sorted(df["frame"].unique())
        T_total = len(frames)
        if T_total < seq_len:
            continue

        frame_to_idx = {f: i for i, f in enumerate(frames)}
        sequence = np.zeros((T_total, N_JOINTS, 2), dtype=np.float32)

        for _, row in df.iterrows():
            t = frame_to_idx[row["frame"]]
            j = part_index.get(row["bp"], -1)
            if j >= 0:
                x, y = row["x"], row["y"]
                sequence[t, j, 0] = 0.0 if np.isnan(x) else float(x)
                sequence[t, j, 1] = 0.0 if np.isnan(y) else float(y)

        normalized = normalize_clip(sequence)

        for start in range(0, T_total - seq_len + 1, step):
            all_clips.append(normalized[start : start + seq_len])

    if not all_clips:
        return np.empty((0, seq_len, N_JOINTS, 2), dtype=np.float32)

    return np.stack(all_clips).astype(np.float32)
