"""
HUMOTO dataset loader: Mixamo GLB/GLTF + YAML → normalized COCO-17 clips.

Directory layout expected:
    humoto_/
    ├── humoto_0805/
    │   ├── carry_frying_pan_both_hands_walk_around-505/
    │   │   ├── carry_frying_pan_both_hands_walk_around-505.glb   ← used
    │   │   ├── carry_frying_pan_both_hands_walk_around-505.fbx   ← ignored
    │   │   └── carry_frying_pan_both_hands_walk_around-505.yaml
    │   └── ...
    └── humoto_objects_0805/
        └── ...  (object meshes, no character animations — skipped)

We read the .glb files (GLTF Binary format) using pygltflib (pure Python, pip only).
Each .glb contains a Mixamo-rigged character; the first animation stores
per-bone translation/rotation keyframes.

Dependencies:
    pip install pygltflib pyyaml
    (no system libraries required)

Mixamo → COCO-17 joint mapping
--------------------------------
    LeftArm      → left_shoulder  (5)
    RightArm     → right_shoulder (6)
    LeftForeArm  → left_elbow     (7)
    RightForeArm → right_elbow    (8)
    LeftHand     → left_wrist     (9)
    RightHand    → right_wrist    (10)
    LeftUpLeg    → left_hip       (11)
    RightUpLeg   → right_hip      (12)
    LeftLeg      → left_knee      (13)
    RightLeg     → right_knee     (14)
    LeftFoot     → left_ankle     (15)
    RightFoot    → right_ankle    (16)
    Head         → nose           (0)
"""

import pathlib
import struct
from typing import Optional

import numpy as np
import yaml

from kinescope.skeleton import COCO_PART_NAMES
from kinescope.pretrain._normalize_clip import normalize_clip

# Mixamo bone name (no prefix) → COCO-17 index
_MIXAMO_TO_COCO = {
    "LeftArm":      5,
    "RightArm":     6,
    "LeftForeArm":  7,
    "RightForeArm": 8,
    "LeftHand":     9,
    "RightHand":    10,
    "LeftUpLeg":    11,
    "RightUpLeg":   12,
    "LeftLeg":      13,
    "RightLeg":     14,
    "LeftFoot":     15,
    "RightFoot":    16,
    "Head":         0,
}

_PREFIXES = ("mixamorig2:", "mixamorig:")
N_JOINTS = len(COCO_PART_NAMES)  # 17


def _strip(name: str) -> str:
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _read_yaml_segments(yaml_path: pathlib.Path) -> list:
    """Parse HUMOTO YAML sidecar → list of (start_frame, end_frame) from long_script."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if "long_script" in data and data["long_script"]:
        return [(seg["start_frame"], seg["end_frame"]) for seg in data["long_script"]]
    return [(data.get("start_frame", 0), data.get("end_frame", None))]


# ── GLTF/GLB reading ─────────────────────────────────────────────────────────

def _accessor_to_array(gltf, accessor_idx: int, binary_blob: bytes) -> np.ndarray:
    """
    Read a GLTF accessor into a numpy array.

    Handles types: SCALAR, VEC2, VEC3, VEC4, MAT4.
    Component types: FLOAT (5126), UNSIGNED_SHORT (5123), UNSIGNED_INT (5125).
    """
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]

    byte_offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count

    _type_size = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    _comp_fmt = {5126: "f", 5123: "H", 5125: "I"}
    _comp_bytes = {5126: 4, 5123: 2, 5125: 4}

    n_comp = _type_size[acc.type]
    fmt = _comp_fmt[acc.componentType]
    comp_bytes = _comp_bytes[acc.componentType]
    stride = bv.byteStride or (n_comp * comp_bytes)

    values = []
    for i in range(count):
        offset = byte_offset + i * stride
        row = struct.unpack_from(f"{n_comp}{fmt}", binary_blob, offset)
        values.append(row)

    arr = np.array(values, dtype=np.float32)
    if n_comp == 1:
        arr = arr.reshape(-1)
    return arr


def _quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """GLTF quaternion [x, y, z, w] → 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y],
    ], dtype=np.float64)


def _glb_to_array(glb_path: pathlib.Path) -> Optional[np.ndarray]:
    """
    Load a Mixamo GLB file and return (T, 17, 3) world-space COCO-17 positions.

    Returns None if the file has no animation or mapped bones.

    Requires: pip install pygltflib
    """
    try:
        from pygltflib import GLTF2
    except ImportError:
        raise ImportError(
            "pygltflib is required for GLB loading.\n"
            "Install with: pip install pygltflib"
        )

    try:
        gltf = GLTF2().load(str(glb_path))
    except Exception:
        return None

    if not gltf.animations:
        return None

    # Binary buffer (GLB embeds one buffer)
    try:
        blob = gltf.binary_blob()
    except Exception:
        return None

    anim = gltf.animations[0]

    # Build node name → index map
    node_name = {i: (n.name or "") for i, n in enumerate(gltf.nodes)}
    node_stripped = {i: _strip(n) for i, n in node_name.items()}

    # Build parent map
    parent = {i: None for i in range(len(gltf.nodes))}
    for i, node in enumerate(gltf.nodes):
        for child in (node.children or []):
            parent[child] = i

    # Node rest-pose local transform (translation only; rotation applied via rest matrix)
    def _node_local_rest(node):
        t = np.array(node.translation or [0, 0, 0], dtype=np.float64)
        q = np.array(node.rotation or [0, 0, 0, 1], dtype=np.float64)
        s = np.array(node.scale or [1, 1, 1], dtype=np.float64)
        R = _quat_to_mat3(q)
        T = np.eye(4)
        T[:3, :3] = R * s[np.newaxis, :]
        T[:3, 3] = t
        return T

    rest_local = [_node_local_rest(n) for n in gltf.nodes]

    # Parse animation channels: {node_idx: {"translation": times+vals, "rotation": t+v}}
    channels_data: dict[int, dict] = {}
    for ch in anim.channels:
        nidx = ch.target.node
        path = ch.target.path
        sampler = anim.samplers[ch.sampler]
        times = _accessor_to_array(gltf, sampler.input, blob)
        values = _accessor_to_array(gltf, sampler.output, blob)
        if nidx not in channels_data:
            channels_data[nidx] = {}
        channels_data[nidx][path] = (times, values)

    if not channels_data:
        return None

    # Determine frame count from the longest time track (assume 30fps)
    all_times = []
    for d in channels_data.values():
        for times, _ in d.values():
            all_times.extend(times.tolist())
    if not all_times:
        return None

    max_t = max(all_times)
    # Infer fps: most common tick spacing
    fps = 30.0
    n_frames = max(1, int(round(max_t * fps)) + 1)

    def _sample_vec3(times, vals, t_sec):
        idx = max(0, np.searchsorted(times, t_sec, side="right") - 1)
        return vals[idx].astype(np.float64)

    def _sample_quat(times, vals, t_sec):
        idx = max(0, np.searchsorted(times, t_sec, side="right") - 1)
        return vals[idx].astype(np.float64)  # [x, y, z, w]

    # Find which node indices map to COCO-17
    coco_nodes = {
        nidx: _MIXAMO_TO_COCO[node_stripped[nidx]]
        for nidx in range(len(gltf.nodes))
        if node_stripped[nidx] in _MIXAMO_TO_COCO
    }

    if not coco_nodes:
        return None

    # Topological order for FK
    def topo_order():
        visited, order = set(), []

        def _visit(i):
            if i in visited:
                return
            visited.add(i)
            p = parent[i]
            if p is not None and p not in visited:
                _visit(p)
            order.append(i)

        for i in coco_nodes:
            _visit(i)
        return order

    order = topo_order()

    out = np.zeros((n_frames, N_JOINTS, 3), dtype=np.float32)

    for frame in range(n_frames):
        t_sec = frame / fps
        world: dict[int, np.ndarray] = {}

        for nidx in order:
            p = parent[nidx]
            parent_world = world.get(p, np.eye(4)) if p is not None else np.eye(4)

            d = channels_data.get(nidx, {})
            # Build local transform from animation (or rest pose)
            if "translation" in d:
                trans = _sample_vec3(*d["translation"], t_sec)
            else:
                trans = rest_local[nidx][:3, 3]

            if "rotation" in d:
                rot = _sample_quat(*d["rotation"], t_sec)
                R = _quat_to_mat3(rot)
            else:
                R = rest_local[nidx][:3, :3]

            local = np.eye(4)
            local[:3, :3] = R
            local[:3, 3] = trans
            world[nidx] = parent_world @ local

        for nidx, coco_idx in coco_nodes.items():
            M = world.get(nidx)
            if M is not None:
                out[frame, coco_idx] = M[:3, 3].astype(np.float32)

    # Eyes and ears (1-4) not in Mixamo; copy nose so normalize_clip sees valid coords
    out[:, 1:5] = out[:, 0:1]

    return out


def _extract_clips_from_sequence(
    sequence: np.ndarray,
    segments: list,
    seq_len: int,
) -> list:
    """Extract fixed-length clips respecting YAML segment boundaries."""
    T = sequence.shape[0]
    clips = []
    step = seq_len // 2

    for seg_start, seg_end in segments:
        if seg_end is None:
            seg_end = T
        seg_end = min(seg_end, T)
        if seg_end - seg_start < seq_len:
            continue
        for start in range(seg_start, seg_end - seq_len + 1, step):
            clips.append(sequence[start : start + seq_len])

    return clips


def load_humoto_clips(
    data_dir: pathlib.Path,
    seq_len: int = 60,
    coord_dim: int = 2,
) -> np.ndarray:
    """
    Load all HUMOTO GLB sequences, convert Mixamo → COCO-17, normalize, and chunk.

    Scans data_dir recursively for *.glb files (skips humoto_objects_* dirs
    which contain object meshes with no character animation).

    Parameters
    ----------
    data_dir : Path — humoto root (e.g., .../humoto_/)
    seq_len : int — clip length in frames
    coord_dim : int — 2 to project XY, 3 to keep XYZ

    Returns
    -------
    (N, seq_len, 17, coord_dim) float32 array
    """
    data_dir = pathlib.Path(data_dir)

    # Collect GLB files; skip the humoto_objects_* directories (no character anim)
    glb_files = sorted(
        p for p in data_dir.rglob("*.glb")
        if "objects" not in p.parts[-3]  # skip humoto_objects_0805 subdir
    )

    if not glb_files:
        raise FileNotFoundError(
            f"No .glb files found in {data_dir} (excluding object dirs).\n"
            f"Expected HUMOTO data at: {data_dir}/humoto_0805/<sequence>/<sequence>.glb"
        )

    all_clips = []
    n_loaded, n_skipped = 0, 0

    for glb_path in glb_files:
        yaml_path = glb_path.with_suffix(".yaml")
        segments = _read_yaml_segments(yaml_path) if yaml_path.exists() else [(0, None)]

        sequence = _glb_to_array(glb_path)  # (T, 17, 3) or None
        if sequence is None:
            n_skipped += 1
            continue

        if coord_dim == 2:
            sequence = sequence[:, :, :2]

        sequence = normalize_clip(sequence)
        clips = _extract_clips_from_sequence(sequence, segments, seq_len)
        all_clips.extend(clips)
        n_loaded += 1

    print(f"HUMOTO: loaded {n_loaded} sequences, skipped {n_skipped}, "
          f"produced {len(all_clips)} clips")

    if not all_clips:
        return np.empty((0, seq_len, N_JOINTS, coord_dim), dtype=np.float32)

    return np.stack(all_clips).astype(np.float32)
