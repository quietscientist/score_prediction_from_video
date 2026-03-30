"""
UDysRS dataset loader: JSON pose exports → COCO-17 (T, 17, 2) clips + scores.

Three tasks, each with its own JSON file and joint schema:

  Drinking / Communication — 15 joints:
    head, neck*, face*, Lsho, Rsho, Lelb, Relb, Lwri, Rwri,
    Lhip, Rhip, Lkne, Rkne, Lank, Rank
    (* neck and face dropped; not in COCO-17)

  LA (leg agility) — 21 joints, temporally split into two halves:
    Upper body (full clip): head, neck*, face*, Lsho, Rsho, Lelb, Relb, Lwri, Rwri
    Half 1 — L leg active, R leg resting:
      Lhip_act, Lkne_act, Lank_act  (L active)
      Rhip_rst, Rkne_rst, Rank_rst  (R resting = same segment)
    Half 2 — R leg active, L leg resting:
      Rhip_act, Rkne_act, Rank_act  (R active)
      Lhip_rst, Lkne_rst, Lank_rst  (L resting = same segment)
    → yielded as TWO separate clips per recording.

COCO-17 mapping:
  head  → 0 (nose proxy)
  Lsho  → 5,  Rsho  → 6
  Lelb  → 7,  Relb  → 8
  Lwri  → 9,  Rwri  → 10
  Lhip  → 11, Rhip  → 12
  Lkne  → 13, Rkne  → 14
  Lank  → 15, Rank  → 16
  joints 1-4 (eyes/ears): NaN — absent in this dataset

Score:
  Sum of 6 UDysRS body-region items per task (0–max varies by task).
  Z-scored across all clips within each task before combining.
"""

import json
import pathlib
from typing import Optional

import numpy as np

N_JOINTS = 17

# ── Joint name → COCO-17 index ────────────────────────────────────────────────
_NAME_TO_COCO = {
    "face":  0,   # face centroid → nose proxy (consistent with other loaders)
    # "head" is crown-of-head; no COCO-17 equivalent — dropped
    "Lsho":  5,  "Rsho":  6,
    "Lelb":  7,  "Relb":  8,
    "Lwri":  9,  "Rwri": 10,
    "Lhip": 11,  "Rhip": 12,
    "Lkne": 13,  "Rkne": 14,
    "Lank": 15,  "Rank": 16,
}

# For LA _act/_rst variants, strip the suffix to get the base name
def _la_name(joint: str) -> Optional[str]:
    """Map LA joint name (with _act/_rst suffix) to COCO index, or None if unmapped."""
    base = joint.replace("_act", "").replace("_rst", "")
    return _NAME_TO_COCO.get(base)


def _pos_to_array(position: dict, joint_map: dict) -> np.ndarray:
    """
    Convert a position dict {joint_name: [[x,y], ...]} to (T, 17, 2) float32.

    joint_map: {joint_name_in_position: coco_index}
    T = length of the first joint in joint_map.
    Unmapped COCO joints are NaN.
    """
    # Determine T from mapped joints
    lengths = [len(position[j]) for j in joint_map if j in position]
    if not lengths:
        return None
    T = min(lengths)

    arr = np.full((T, N_JOINTS, 2), np.nan, dtype=np.float32)
    for jname, cidx in joint_map.items():
        if jname in position and len(position[jname]) >= T:
            arr[:T, cidx] = np.array(position[jname][:T], dtype=np.float32)
    return arr


# ── Per-task loaders ──────────────────────────────────────────────────────────

def _load_drinking_or_comm(pose_file: pathlib.Path, score_dict: dict, sn: dict) -> list:
    """
    Returns list of dicts: {clip_id, task, array (T,17,2), score_raw, subject_id}
    """
    with open(pose_file) as f:
        data = json.load(f)

    joint_map = {k: v for k, v in _NAME_TO_COCO.items()}  # all 13 joints

    records = []
    for clip_id, clip_data in data.items():
        base = clip_id.split("-")[0].split(" ")[0]
        score_list = (score_dict.get(clip_id)
                      or score_dict.get(base)
                      or (score_dict.get(str(sn[clip_id])) if clip_id in sn else None)
                      or (score_dict.get(str(sn[base])) if base in sn else None))
        if score_list is None:
            continue

        arr = _pos_to_array(clip_data["position"], joint_map)
        if arr is None or len(arr) < 30:
            continue

        records.append({
            "clip_id": clip_id,
            "subject_id": sn.get(clip_id, sn.get(base, clip_id)),
            "array": arr,
            "score_raw": float(sum(score_list)),
        })
    return records


def _load_la(pose_file: pathlib.Path, score_dict: dict, sn: dict) -> list:
    """
    LA (leg agility): yields TWO clips per recording — one per active leg.

    Half 1: upper body [0 : L_act_len] + L_act lower + R_rst lower
    Half 2: upper body [L_act_len : end] + R_act lower + L_rst lower
    """
    with open(pose_file) as f:
        data = json.load(f)

    upper_joints = {j: _NAME_TO_COCO[j]
                    for j in ("face", "Lsho", "Rsho", "Lelb", "Relb", "Lwri", "Rwri")}

    records = []
    for clip_id, clip_data in data.items():
        base = clip_id.split(" ")[0]
        score_list = (score_dict.get(clip_id)
                      or score_dict.get(base)
                      or (score_dict.get(str(sn[clip_id])) if clip_id in sn else None)
                      or (score_dict.get(str(sn[base])) if base in sn else None))
        if score_list is None:
            continue

        pos = clip_data["position"]
        score_raw = float(sum(score_list))

        subj = sn.get(clip_id, sn.get(base, clip_id))

        # Frame counts
        la_len = len(pos.get("Lank_act", []))
        ra_len = len(pos.get("Rank_act", []))
        ub_len = len(pos.get("face", []))
        if la_len == 0 or ra_len == 0 or ub_len == 0:
            continue

        for half, (T, ub_start, act_L, act_R, rst_L, rst_R) in enumerate([
            # Half 1: L active (first la_len frames of upper body)
            (la_len, 0,      "Lhip_act", "Rhip_act", "Lkne_act", "Rkne_act"),
            # Half 2: R active (next ra_len frames of upper body)
            (ra_len, la_len, "Rhip_act", "Lhip_act", "Rkne_act", "Lkne_act"),
        ]):
            # Build joint map for this half
            # Active leg: L for half 0, R for half 1
            if half == 0:
                lower_map = {
                    "Lhip_act": 11, "Rhip_rst": 12,
                    "Lkne_act": 13, "Rkne_rst": 14,
                    "Lank_act": 15, "Rank_rst": 16,
                }
            else:
                lower_map = {
                    "Rhip_act": 12, "Lhip_rst": 11,
                    "Rkne_act": 14, "Lkne_rst": 13,
                    "Rank_act": 16, "Lank_rst": 15,
                }

            arr = np.full((T, N_JOINTS, 2), np.nan, dtype=np.float32)

            # Upper body
            ub_end = ub_start + T
            for jname, cidx in upper_joints.items():
                if jname in pos and len(pos[jname]) >= ub_end:
                    arr[:, cidx] = np.array(pos[jname][ub_start:ub_end], dtype=np.float32)

            # Lower body
            for jname, cidx in lower_map.items():
                if jname in pos and len(pos[jname]) >= T:
                    arr[:, cidx] = np.array(pos[jname][:T], dtype=np.float32)

            if T < 30:
                continue

            records.append({
                "clip_id": f"{clip_id}_half{half}",
                "subject_id": subj,
                "array": arr,
                "score_raw": score_raw,
                "la_half": half,
            })

    return records


# ── Public API ────────────────────────────────────────────────────────────────

def load_udysrs(data_dir: str) -> dict:
    """
    Load all three UDysRS tasks and return z-scored clips ready for linear probe.

    Parameters
    ----------
    data_dir : str — path to UDysRS_UPDRS_Export/ or inner UDysRS_UPDRS_Export/

    Returns
    -------
    dict with keys:
      'arrays'   : (N, T, 17, 2) list of variable-length arrays (ragged — not stacked)
      'scores'   : (N,) float32 array of z-scored UDysRS totals
      'scores_raw': (N,) float32 array of raw summed scores
      'task'     : (N,) list of str — 'la', 'drinking', 'communication'
      'clip_ids' : (N,) list of str
      'subject_ids': (N,) list
    """
    data_dir = pathlib.Path(data_dir)
    # Handle both outer and inner directory
    inner = data_dir / "UDysRS_UPDRS_Export"
    if inner.exists():
        data_dir = inner

    with open(data_dir / "UDysRS.txt") as f:
        udysrs = json.load(f)
    with open(data_dir / "sn_numbers.txt") as f:
        sn = json.load(f)

    all_records = []

    for task_name, pose_file, score_key, loader_fn in [
        ("drinking",      data_dir / "Drinking_all_export.txt",      "Drinking",      _load_drinking_or_comm),
        ("communication", data_dir / "Communication_all_export.txt",  "Communication", _load_drinking_or_comm),
        ("la",            data_dir / "LA_split_all_export.txt",       "Higher",        _load_la),
    ]:
        records = loader_fn(pose_file, udysrs[score_key], sn)
        for r in records:
            r["task"] = task_name
        all_records.extend(records)
        print(f"{task_name}: {len(records)} clips")

    # Z-score within each task
    from collections import defaultdict
    task_scores = defaultdict(list)
    for r in all_records:
        task_scores[r["task"]].append(r["score_raw"])

    task_stats = {}
    for task, vals in task_scores.items():
        mu, sigma = np.mean(vals), np.std(vals)
        task_stats[task] = (mu, sigma if sigma > 1e-6 else 1.0)

    scores_z = np.array([
        (r["score_raw"] - task_stats[r["task"]][0]) / task_stats[r["task"]][1]
        for r in all_records
    ], dtype=np.float32)

    print(f"Total: {len(all_records)} clips")

    return {
        "arrays":      [r["array"] for r in all_records],
        "scores":      scores_z,
        "scores_raw":  np.array([r["score_raw"] for r in all_records], dtype=np.float32),
        "task":        [r["task"] for r in all_records],
        "clip_ids":    [r["clip_id"] for r in all_records],
        "subject_ids": [r["subject_id"] for r in all_records],
    }
