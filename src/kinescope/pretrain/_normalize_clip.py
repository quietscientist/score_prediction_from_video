"""
Numpy-native skeleton normalization for pretraining data loaders.

Replicates normalise_skeletons() from processing/normalization.py but operates
directly on (T, 17, 2) numpy arrays without requiring a pandas DataFrame.

Used by amass_loader, ntu_loader, fbx_loader, coco_loader, and the linear probe.

Normalization strategy
----------------------
Translation:  centered on the CLIP-MEAN shoulder/hip midpoint (not per-frame).
              This preserves whole-body sway — the temporal variation of the body
              center — which is clinically relevant for conditions like dyskinesia.
              Per-frame centering would subtract out this signal entirely.
Rotation:     per-frame shoulder/hip angle, rotated to canonical upright orientation.
              Removes camera angle and within-frame body tilt.
Scale:        per-frame trunk length (shoulder-center to hip-center distance).
              Removes subject height and distance-to-camera variation.
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

    Steps:
      1. Compute clip-mean shoulder/hip midpoints as translation reference.
         Using the clip mean (not per-frame) preserves whole-body sway —
         the temporal displacement of the body center relative to its average
         position — which is a clinically relevant signal for movement disorders.
      2. Per-frame: compute shoulder/hip orientation angles and rotate to align
         the body axis vertically.
      3. Per-frame: scale by trunk length (shoulder-center to hip-center distance).
      4. Shift lower body so trunk bottom = y=1, trunk top = y=0.

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

    # Clip-mean midpoints for translation — preserves sway across frames
    l_sh_all = clip[:, _L_SHOULDER, :]  # (T, 2)
    r_sh_all = clip[:, _R_SHOULDER, :]
    l_hp_all = clip[:, _L_HIP, :]
    r_hp_all = clip[:, _R_HIP, :]
    mean_uref = np.nanmean((l_sh_all + r_sh_all) / 2.0, axis=0)  # (2,)
    mean_lref = np.nanmean((l_hp_all + r_hp_all) / 2.0, axis=0)  # (2,)

    out = np.zeros_like(clip)

    for t in range(T):
        frame = clip[t]  # (17, 2)

        l_sh = frame[_L_SHOULDER]
        r_sh = frame[_R_SHOULDER]
        l_hp = frame[_L_HIP]
        r_hp = frame[_R_HIP]

        uref_t = (l_sh + r_sh) / 2.0   # per-frame shoulder midpoint (rotation ref)
        lref_t = (l_hp + r_hp) / 2.0   # per-frame hip midpoint (rotation ref)

        trunk_len = np.linalg.norm(uref_t - lref_t)
        if trunk_len < 1e-6:
            out[t] = frame  # can't normalize; keep original
            continue

        # Per-frame orientation angles (rotation + scale remain frame-local)
        sh_angle = np.arctan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0])
        hp_angle = np.arctan2(r_hp[1] - l_hp[1], r_hp[0] - l_hp[0])

        sh_angle_adj = _adjust_angle(np.array([sh_angle]))[0]
        hp_angle_adj = _adjust_angle(np.array([hp_angle]))[0]

        # Translate relative to clip-mean midpoint (sway preserved),
        # then rotate by per-frame angle and scale by per-frame trunk length
        ca_u, sa_u = np.cos(sh_angle_adj), np.sin(sh_angle_adj)
        ca_l, sa_l = np.cos(hp_angle_adj), np.sin(hp_angle_adj)

        for j in _UPPER:
            dx = frame[j, 0] - mean_uref[0]
            dy = frame[j, 1] - mean_uref[1]
            out[t, j, 0] = (ca_u * dx - sa_u * dy) / trunk_len
            out[t, j, 1] = (sa_u * dx + ca_u * dy) / trunk_len

        for j in _LOWER:
            dx = frame[j, 0] - mean_lref[0]
            dy = frame[j, 1] - mean_lref[1]
            out[t, j, 0] = (ca_l * dx - sa_l * dy) / trunk_len
            out[t, j, 1] = (sa_l * dx + ca_l * dy) / trunk_len + 1.0

    return out
