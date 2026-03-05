"""
Blender script: batch-export HUMOTO FBX joint positions → COCO-17 CSVs.

Usage (no sudo required — Blender reads FBX natively):
    blender --background --python scripts/humoto_to_csv.py -- <humoto_dir> [<output_dir>]

Arguments (after --):
    humoto_dir   : directory containing *.fbx files (HUMOTO dataset)
    output_dir   : where to write *.csv files (default: humoto_dir/coco_csv/)

Output:
    One CSV per FBX file, in kinescope COCO-17 format:
        frame, x, y, bp, confidence
    Ready for use with:
        kinescope pretrain --datasets coco --data-root <parent_of_coco_csv>

Mixamo → COCO-17 joint mapping used:
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

Requires: Blender 3.x or 4.x (free, blender.org)
"""

import csv
import pathlib
import sys

# ── Mixamo bone name (no prefix) → COCO-17 name ──────────────────────────────
_MIXAMO_TO_COCO_NAME = {
    "LeftArm":      "left_shoulder",
    "RightArm":     "right_shoulder",
    "LeftForeArm":  "left_elbow",
    "RightForeArm": "right_elbow",
    "LeftHand":     "left_wrist",
    "RightHand":    "right_wrist",
    "LeftUpLeg":    "left_hip",
    "RightUpLeg":   "right_hip",
    "LeftLeg":      "left_knee",
    "RightLeg":     "right_knee",
    "LeftFoot":     "left_ankle",
    "RightFoot":    "right_ankle",
    "Head":         "nose",
}

_PREFIXES = ("mixamorig2:", "mixamorig:")


def _strip(name: str) -> str:
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _export_fbx(fbx_path: pathlib.Path, out_csv: pathlib.Path) -> int:
    """
    Import one FBX into Blender, sample bone world positions for every frame,
    write a COCO-17 CSV. Returns number of frames written (0 on failure).
    """
    import bpy  # available inside Blender's Python

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX
    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=True)
    except Exception as e:
        print(f"  [SKIP] Could not import {fbx_path.name}: {e}")
        return 0

    # Find armature
    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature is None:
        print(f"  [SKIP] No armature in {fbx_path.name}")
        return 0

    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    n_frames = frame_end - frame_start + 1

    rows = []

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        # Update dependency graph so pose matrices are current
        bpy.context.view_layer.update()

        for bone in armature.pose.bones:
            stripped = _strip(bone.name)
            coco_name = _MIXAMO_TO_COCO_NAME.get(stripped)
            if coco_name is None:
                continue

            # World-space head position of the bone
            world_pos = armature.matrix_world @ bone.head
            rows.append({
                "frame": frame - frame_start,  # 0-indexed
                "x": world_pos.x,
                "y": world_pos.y,
                "bp": coco_name,
                "confidence": 1.0,
            })

    if not rows:
        print(f"  [SKIP] No mapped bones found in {fbx_path.name}")
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "x", "y", "bp", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    return n_frames


def main():
    # Parse arguments after '--'
    argv = sys.argv
    try:
        sep = argv.index("--")
        args = argv[sep + 1:]
    except ValueError:
        args = []

    if not args:
        print(__doc__)
        sys.exit(1)

    humoto_dir = pathlib.Path(args[0]).expanduser().resolve()
    if len(args) >= 2:
        out_dir = pathlib.Path(args[1]).expanduser().resolve()
    else:
        out_dir = humoto_dir / "coco_csv"

    fbx_files = sorted(humoto_dir.rglob("*.fbx"))
    if not fbx_files:
        print(f"No .fbx files found in {humoto_dir}")
        sys.exit(1)

    print(f"Converting {len(fbx_files)} FBX files → {out_dir}/")

    ok, skipped = 0, 0
    for fbx_path in fbx_files:
        out_csv = out_dir / (fbx_path.stem + ".csv")
        print(f"  {fbx_path.name} ...", end=" ", flush=True)
        n = _export_fbx(fbx_path, out_csv)
        if n > 0:
            print(f"{n} frames → {out_csv.name}")
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} converted, {skipped} skipped.")
    print(f"\nOutput CSVs in: {out_dir}")
    print(f"\nNext step — place in your data dir and pretrain:")
    print(f"  mkdir -p ~/kinescope_data/coco")
    print(f"  cp {out_dir}/*.csv ~/kinescope_data/coco/")
    print(f"  kinescope pretrain --datasets coco --output ./ckpt --artifacts-dir ./artifacts")


if __name__ == "__main__":
    main()
