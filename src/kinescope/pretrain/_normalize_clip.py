"""
Numpy-native skeleton normalization for pretraining data loaders.

Replicates normalise_skeletons() from processing/normalization.py but operates
directly on (T, 17, 2) numpy arrays without requiring a pandas DataFrame.

Used by amass_loader, ntu_loader, fbx_loader, and coco_loader.
"""

import numpy as np

# COCO-17 joint indices used for normalization reference axes
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12

# Upper body joint indices (use shoulder midpoint + shoulder angle as reference)
_UPPER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Lower body joint indices (use hip midpoint + hip angle as reference)
_LOWER = np.array([11, 12, 13, 14, 15, 16])


def _adjust_angle(angle: np.ndarray) -> np.ndarray:
    """
    Map raw shoulder/hip angle to a canonical upward-pointing reference direction.

    Replicates the angle adjustment in normalise_skeletons():
        if angle < 0: angle += 2π
        if angle < π: angle = π - angle
        if π < angle < 2π: angle = 3π - angle
    """
    a = angle.copy()
    a[a < 0] += 2 * np.pi
    mask_lt_pi = a < np.pi
    a[mask_lt_pi] = np.pi - a[mask_lt_pi]
    mask_gt_pi = (a > np.pi) & (a < 2 * np.pi)
    a[mask_gt_pi] = 3 * np.pi - a[mask_gt_pi]
    return a


def normalize_clip(clip: np.ndarray) -> np.ndarray:
    """
    Normalize a COCO-17 pose clip to body-relative coordinates.

    Replicates normalise_skeletons() per-frame:
      1. Compute shoulder/hip midpoints (uref, lref)
      2. Compute shoulder/hip orientation angles
      3. Rotate upper/lower body separately to align body axis vertically
      4. Scale by trunk length (shoulder-center to hip-center distance)
      5. Shift lower body so trunk bottom = y=1, trunk top = y=0

    Parameters
    ----------
    clip : (T, J, 2) float32 array — raw pixel or world-space COCO-17 coordinates
        J must be 17. Only 2D (x, y) supported; for 3D input project to 2D first.

    Returns
    -------
    (T, J, 2) float32 array — normalized body-relative coordinates
    """
    clip = np.asarray(clip, dtype=np.float32)
    T, J, D = clip.shape
    assert D == 2, "normalize_clip requires 2D input (T, J, 2). Project 3D to 2D first."
    assert J == 17, f"Expected 17 COCO joints, got {J}"

    out = np.zeros_like(clip)

    for t in range(T):
        frame = clip[t]  # (17, 2)

        l_sh = frame[_L_SHOULDER]  # left_shoulder
        r_sh = frame[_R_SHOULDER]  # right_shoulder
        l_hp = frame[_L_HIP]       # left_hip
        r_hp = frame[_R_HIP]       # right_hip

        uref = (l_sh + r_sh) / 2.0   # shoulder midpoint
        lref = (l_hp + r_hp) / 2.0   # hip midpoint

        trunk_len = np.linalg.norm(uref - lref)
        if trunk_len < 1e-6:
            out[t] = frame  # can't normalize; keep original
            continue

        # Reference angles
        sh_angle = np.arctan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0])
        hp_angle = np.arctan2(r_hp[1] - l_hp[1], r_hp[0] - l_hp[0])

        sh_angle_adj = _adjust_angle(np.array([sh_angle]))[0]
        hp_angle_adj = _adjust_angle(np.array([hp_angle]))[0]

        # Normalize upper body joints
        for j in _UPPER:
            dx = frame[j, 0] - uref[0]
            dy = frame[j, 1] - uref[1]
            ca, sa = np.cos(sh_angle_adj), np.sin(sh_angle_adj)
            out[t, j, 0] = (ca * dx - sa * dy) / trunk_len
            out[t, j, 1] = (sa * dx + ca * dy) / trunk_len

        # Normalize lower body joints (shift y by +1 so trunk bottom = 1)
        for j in _LOWER:
            dx = frame[j, 0] - lref[0]
            dy = frame[j, 1] - lref[1]
            ca, sa = np.cos(hp_angle_adj), np.sin(hp_angle_adj)
            out[t, j, 0] = (ca * dx - sa * dy) / trunk_len
            out[t, j, 1] = (sa * dx + ca * dy) / trunk_len + 1.0

    return out
