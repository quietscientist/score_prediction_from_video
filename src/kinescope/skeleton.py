# COCO-17 keypoint definitions
# Standard COCO pose challenge format (17 body parts)

COCO_PART_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Internal Neck keypoint name (not in COCO-17; computed as shoulder midpoint for normalization)
NECK = "neck"

# Limb connections for visualization (pairs of part indices)
COCO_LIMBS = [
    (0, 1),   # Nose → Left Eye
    (0, 2),   # Nose → Right Eye
    (1, 3),   # Left Eye → Left Ear
    (2, 4),   # Right Eye → Right Ear
    (5, 7),   # Left Shoulder → Left Elbow
    (7, 9),   # Left Elbow → Left Wrist
    (6, 8),   # Right Shoulder → Right Elbow
    (8, 10),  # Right Elbow → Right Wrist
    (5, 6),   # Left Shoulder → Right Shoulder
    (5, 11),  # Left Shoulder → Left Hip
    (6, 12),  # Right Shoulder → Right Hip
    (11, 12), # Left Hip → Right Hip
    (11, 13), # Left Hip → Left Knee
    (13, 15), # Left Knee → Left Ankle
    (12, 14), # Right Hip → Right Knee
    (14, 16), # Right Knee → Right Ankle
]

COCO_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170],
]

# Index lookup: name → index
COCO_PART_INDEX = {name: i for i, name in enumerate(COCO_PART_NAMES)}

# Joints used in kinematic feature extraction
LIMB_JOINTS = ["left_ankle", "right_ankle", "left_wrist", "right_wrist",
               "left_elbow", "right_elbow", "left_knee", "right_knee"]

# Joint angle definitions: (vertex, proximal, distal) — vertex is the joint being measured
JOINT_ANGLE_TRIPLETS = [
    # Shoulder abduction (vertex=shoulder, proximal=neck, distal=elbow)
    ("right_shoulder", NECK, "right_elbow"),
    ("left_shoulder",  NECK, "left_elbow"),
    # Elbow flexion (vertex=elbow, proximal=shoulder, distal=wrist)
    ("right_elbow", "right_shoulder", "right_wrist"),
    ("left_elbow",  "left_shoulder",  "left_wrist"),
    # Hip abduction (vertex=hip, proximal=opposite_hip, distal=knee)
    ("right_hip", "left_hip", "right_knee"),
    ("left_hip",  "right_hip", "left_knee"),
    # Knee flexion (vertex=knee, proximal=hip, distal=ankle)
    ("right_knee", "right_hip", "right_ankle"),
    ("left_knee",  "left_hip",  "left_ankle"),
]
