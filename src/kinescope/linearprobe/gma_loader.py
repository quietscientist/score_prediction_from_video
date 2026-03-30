"""
GMA (General Movements Assessment) dataset loader.

Pose format: per-video JSON files with naming convention:
  {infant_id}_{session}_{age_code}.json
  e.g.  73_2_1.json  (age_code: 0=term, 1=3-4 month fidgety window)

Filters applied:
  - age_code == 1 (3-4 month fidgety window only)
  - For infants with multiple sessions, keep only the last (highest session number)

Instance selection (multi-person frames):
  - Primary: highest mean keypoint confidence score across all 17 joints
  - Tiebreaker: largest bounding box area (infant is main subject)

COCO-17 keypoints used directly.

Score (gma_scores.csv):
  1 = F+   normal fidgety present
  2 = F+/- sporadic fidgety
  3 = F-   absent fidgety (abnormal)
  Binary label: score >= 2 → abnormal (1), else normal (0)

Usage:
    from kinescope.linearprobe.gma_loader import load_gma
    data = load_gma("/path/to/gma/")
"""

import csv
import json
import pathlib
import re
from collections import defaultdict
from typing import Optional

import numpy as np

N_JOINTS = 17
AGE_CATEGORIES = {1}  # 0=term, 1=3-4 month fidgety window


def _parse_filename(stem: str) -> Optional[tuple]:
    """
    Parse {infant}_{session}_{age} from filename stem.
    Also handles legacy 'video_{id}' format (sample files).
    Returns (infant_id, session, age_category) or None if unparseable.
    """
    # Legacy sample format: video_000073
    m = re.fullmatch(r"video_(\d+)", stem)
    if m:
        return (str(int(m.group(1))), 0, 1)  # assume age_code=1 (3-4 month) for sample

    # Real format: {infant}_{session}_{age_code}
    parts = stem.split("_")
    if len(parts) == 3:
        try:
            infant_id = parts[0]
            session   = int(parts[1])
            age       = int(parts[2])
            return (infant_id, session, age)
        except ValueError:
            pass

    return None


def _select_instance(instances: list) -> Optional[dict]:
    """
    Pick the infant from a multi-instance frame.
    Primary: highest mean keypoint confidence.
    Tiebreaker: largest bounding box area.
    """
    if not instances:
        return None
    if len(instances) == 1:
        return instances[0]

    def _score(inst):
        kp_mean = float(np.mean(inst["keypoint_scores"]))
        bbox = inst["bbox"][0]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return (kp_mean, area)

    return max(instances, key=_score)


def _json_to_array(frames: list) -> np.ndarray:
    """Convert list of frame dicts → (T, 17, 2) float32. Low-confidence joints → NaN."""
    T = len(frames)
    arr = np.full((T, N_JOINTS, 2), np.nan, dtype=np.float32)
    for t, frame in enumerate(frames):
        inst = _select_instance(frame.get("instances", []))
        if inst is None:
            continue
        for j, (xy, conf) in enumerate(zip(inst["keypoints"], inst["keypoint_scores"])):
            if conf > 0.0:
                arr[t, j, 0] = float(xy[0])
                arr[t, j, 1] = float(xy[1])
    return arr


def load_gma(data_dir: str, scores_file: Optional[str] = None,
             all_sessions: bool = False) -> dict:
    """
    Load GMA dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing JSON pose files ({infant}_{session}_{age_code}.json).
    scores_file : str, optional
        Path to gma_scores.csv.  Defaults to {data_dir}/gma_scores.csv.

    Returns
    -------
    dict with keys:
      'arrays'        : list of (T, 17, 2) float32 arrays (variable T)
      'scores_raw'    : (N,) int32 — GMA score (1/2/3)
      'scores'        : (N,) float32 — z-scored raw score
      'binary'        : (N,) int32 — 0=normal (F+), 1=abnormal (F+/- or F-)
      'subject_ids'   : list of str — infant id
      'age_corrected' : (N,) float32 — corrected gestational age (weeks)
    """
    data_dir  = pathlib.Path(data_dir)
    pose_dir  = data_dir
    scores_file = pathlib.Path(scores_file) if scores_file else data_dir / "gma_scores.csv"

    # ── Load scores ────────────────────────────────────────────────────────────
    scores_map = {}
    with open(scores_file) as f:
        for row in csv.DictReader(f):
            if not row["score"]:
                continue
            scores_map[row["infant"]] = {
                "score": int(row["score"]),
                "age_corrected": float(row["age_corrected"]) if row["age_corrected"] else float("nan"),
            }

    # ── Discover pose files, filter age, keep last session per infant ──────────
    all_files = sorted(pose_dir.glob("*.json"))
    if not all_files:
        raise FileNotFoundError(f"No .json files in {pose_dir}")

    # Group by infant_id → {session: path}
    by_infant: defaultdict = defaultdict(dict)
    skipped_parse, skipped_age = 0, 0
    for pf in all_files:
        parsed = _parse_filename(pf.stem)
        if parsed is None:
            skipped_parse += 1
            continue
        infant_id, session, age = parsed
        if age not in AGE_CATEGORIES:
            skipped_age += 1
            continue
        by_infant[infant_id][session] = pf

    # Keep last session per infant (default) or all sessions
    if all_sessions:
        selected_list = [(iid, sess, pf)
                         for iid, sessions in by_infant.items()
                         for sess, pf in sessions.items()]
        print(f"Found {len(selected_list)} recordings from "
              f"{len(by_infant)} infants at 3-4 months "
              f"({skipped_age} files skipped: wrong age, {skipped_parse} unparseable)")
    else:
        selected_list = [(iid, max(sessions), sessions[max(sessions)])
                         for iid, sessions in by_infant.items()]
        print(f"Found {len(selected_list)} infants at 3-4 months (age_code=1) "
              f"({skipped_age} files skipped: wrong age, {skipped_parse} unparseable)")

    # ── Load arrays ────────────────────────────────────────────────────────────
    records = []
    skipped_score, skipped_short = 0, 0
    for infant_id, _session, pf in sorted(selected_list):
        if infant_id not in scores_map:
            skipped_score += 1
            continue
        with open(pf) as f:
            frames = json.load(f)
        arr = _json_to_array(frames)
        if arr.shape[0] < 30:
            skipped_short += 1
            continue
        records.append({"subject_id": infant_id, "array": arr, **scores_map[infant_id]})

    print(f"Loaded {len(records)} clips "
          f"({skipped_score} skipped: no score, {skipped_short} too short)")

    if not records:
        raise ValueError(f"No valid clips loaded from {data_dir}")

    raw = np.array([r["score"] for r in records], dtype=np.float32)
    mu, sigma = raw.mean(), raw.std()
    scores_z = (raw - mu) / (sigma if sigma > 1e-6 else 1.0)

    from collections import Counter
    dist = Counter(int(r["score"]) for r in records)
    print(f"Score distribution: {dict(sorted(dist.items()))}  "
          f"(1=F+ normal, 2=F+/- sporadic, 3=F- absent)")

    return {
        "arrays":        [r["array"] for r in records],
        "scores_raw":    raw.astype(np.int32),
        "scores":        scores_z,
        "binary":        (raw >= 2).astype(np.int32),
        "subject_ids":   [r["subject_id"] for r in records],
        "age_corrected": np.array([r["age_corrected"] for r in records], dtype=np.float32),
    }
