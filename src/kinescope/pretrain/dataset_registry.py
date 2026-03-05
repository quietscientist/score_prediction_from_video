"""
Dataset registry for Pose-JEPA pretraining.

Resolves dataset paths from $KINESCOPE_DATA_DIR (default: ~/kinescope_data/).

Actual directory layout at /home/msegado/tapedeck/msegado/Datasets/pretraining/
(set KINESCOPE_DATA_DIR to this path):

    pretraining/
    ├── AMASS/                         → dataset name "amass"
    │   ├── CMU/, MPI_HDM05/, ...      (standard AMASS sub-datasets)
    ├── humoto_/                       → dataset name "humoto"
    │   ├── humoto_0805/               (character animations, .glb + .yaml)
    │   └── humoto_objects_0805/       (object meshes, skipped automatically)
    ├── nturgb+d_skeletons/            → dataset name "ntu120" (both dirs combined)
    ├── nturgbd_skeletons_s018_to_s032/→ dataset name "ntu120" (both dirs combined)
    └── idea400_video/                 → raw MP4 videos; needs pose estimation first

Usage
-----
    export KINESCOPE_DATA_DIR=/home/msegado/tapedeck/msegado/Datasets/pretraining

    from kinescope.pretrain.dataset_registry import load_clips
    clips = load_clips(["amass", "humoto", "ntu120"], seq_len=60)

Or pass data_root explicitly:
    clips = load_clips(["amass"], seq_len=60,
                       data_root="/home/msegado/tapedeck/msegado/Datasets/pretraining")
"""

import os
import pathlib
from typing import Optional

import numpy as np

_DEFAULT_ROOT = pathlib.Path.home() / "kinescope_data"

# Maps dataset name → subdirectory name(s) within the data root.
# Value is either a str (single dir) or list of str (multiple dirs, all scanned).
KNOWN_DATASETS = {
    "amass":   "AMASS",
    "humoto":  "humoto_",
    "ntu120":  ["nturgbd_skeletons_s018_to_s032", "nturgb+d_skeletons"],
    "coco":    "coco",
}


def get_data_root(data_root: Optional[str] = None) -> pathlib.Path:
    """
    Return the dataset root directory.

    Priority: explicit data_root arg > $KINESCOPE_DATA_DIR > ~/kinescope_data/
    """
    if data_root is not None:
        return pathlib.Path(data_root)
    env = os.environ.get("KINESCOPE_DATA_DIR")
    if env:
        return pathlib.Path(env)
    return _DEFAULT_ROOT


def get_dataset_paths(name: str, data_root: Optional[str] = None) -> list:
    """
    Return list of existing paths for a named dataset.

    Raises ValueError for unknown names, FileNotFoundError if none exist.
    """
    if name not in KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(KNOWN_DATASETS)}")

    root = get_data_root(data_root)
    subdirs = KNOWN_DATASETS[name]
    if isinstance(subdirs, str):
        subdirs = [subdirs]

    paths = [root / s for s in subdirs if (root / s).exists()]
    if not paths:
        tried = [str(root / s) for s in subdirs]
        raise FileNotFoundError(
            f"No directory found for dataset '{name}'.\n"
            f"Tried: {tried}\n"
            f"Set $KINESCOPE_DATA_DIR to your pretraining data root."
        )
    return paths


def load_clips(
    datasets: list,
    seq_len: int = 60,
    data_root: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load and concatenate clips from one or more datasets.

    Parameters
    ----------
    datasets : list of str — any of: amass, humoto, ntu120, coco
    seq_len : int — clip length in frames
    data_root : str or None — override data root
    verbose : bool

    Returns
    -------
    (N, seq_len, 17, 2) float32 array
    """
    all_clips = []

    for name in datasets:
        paths = get_dataset_paths(name, data_root)
        if verbose:
            print(f"Loading {name} from {[str(p) for p in paths]} ...")

        if name == "amass":
            from kinescope.pretrain.amass_loader import load_amass_clips
            clips_list = [load_amass_clips(p, seq_len=seq_len) for p in paths]
            clips = np.concatenate(clips_list, axis=0) if len(clips_list) > 1 else clips_list[0]

        elif name == "humoto":
            from kinescope.pretrain.fbx_loader import load_humoto_clips
            # humoto_ has humoto_0805 and humoto_objects_0805 inside;
            # load_humoto_clips handles the filtering internally.
            clips_list = [load_humoto_clips(p, seq_len=seq_len) for p in paths]
            clips = np.concatenate(clips_list, axis=0) if len(clips_list) > 1 else clips_list[0]

        elif name == "ntu120":
            from kinescope.pretrain.ntu_loader import load_ntu_clips
            # Both NTU dirs are scanned and combined
            clips_list = [load_ntu_clips(p, seq_len=seq_len) for p in paths]
            clips = np.concatenate([c for c in clips_list if len(c) > 0], axis=0)

        elif name == "coco":
            from kinescope.pretrain.coco_loader import load_coco_clips
            clips_list = [load_coco_clips(p, seq_len=seq_len) for p in paths]
            clips = np.concatenate(clips_list, axis=0) if len(clips_list) > 1 else clips_list[0]

        else:
            raise ValueError(f"No loader for dataset '{name}'")

        if verbose:
            print(f"  → {len(clips)} clips")
        all_clips.append(clips)

    if not all_clips:
        raise ValueError("No datasets loaded.")

    combined = np.concatenate(all_clips, axis=0)
    if verbose:
        print(f"Total clips: {len(combined)}")
    return combined
