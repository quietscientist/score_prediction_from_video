"""PyTorch Dataset for COCO-17 normalized pose clips.

Wraps any source of pose sequences into a clip dataset with optional
motion-aware sampling weights (static clips downweighted to reduce
the "predict same pose" shortcut during Pose-JEPA pretraining).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ClipDataset(Dataset):
    """
    Dataset of fixed-length COCO-17 pose clips.

    Parameters
    ----------
    clips : array-like, shape (N, T, 17, 2) or (N, T, 17, 3)
        Normalized pose clips (after normalise_skeletons())
    motion_weights : array-like of shape (N,), optional
        Per-clip sampling weights. If None, uniform sampling is used.
    """

    def __init__(self, clips, motion_weights=None):
        # Keep as numpy (supports memmap arrays — data is paged in on access).
        self.clips = clips if isinstance(clips, np.ndarray) else np.asarray(clips)
        self.motion_weights = (
            np.asarray(motion_weights, dtype=np.float32)
            if motion_weights is not None
            else None
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        # Defensive sanitization: cached arrays from mixed datasets can contain
        # NaN/Inf outliers that would otherwise poison the JEPA loss.
        clip = np.array(self.clips[idx], dtype=np.float32, copy=True)
        clip = np.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(clip, dtype=torch.float32)

    @staticmethod
    def compute_motion_weights(
        clips,
        static_weight: float = 0.25,
        motion_threshold: float = 0.05,
    ) -> np.ndarray:
        """
        Compute per-clip sampling weights based on motion amount.

        Static clips (mean joint displacement < motion_threshold) are downweighted
        to static_weight. Dynamic clips receive weight 1.0.

        Parameters
        ----------
        clips : (N, T, J, D) array
        static_weight : float — weight for static clips (0 < static_weight <= 1.0)
        motion_threshold : float — displacement threshold in normalized coordinates

        Returns
        -------
        weights : (N,) float array
        """
        clips = np.asarray(clips)
        displacements = np.linalg.norm(np.diff(clips, axis=1), axis=-1)  # (N, T-1, J)
        mean_disp = displacements.mean(axis=(1, 2))  # (N,)
        return np.where(mean_disp > motion_threshold, 1.0, static_weight).astype(
            np.float32
        )
