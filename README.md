# kinescope

**Predict clinical and sports scores from human movement video using kinematics.**

`kinescope` is an open-source Python package for extracting kinematic features from pose estimation data and predicting scores (clinical assessments, sports performance, etc.) using those features.

**Published paper (GigaScience 2025):** see the [`gigascience-2025`](../../tree/gigascience-2025) branch for the exact code used in the paper.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14042732.svg)](https://doi.org/10.5281/zenodo.14042732)

---

## Overview

```
Video
  ↓  (sam3d-video-pose or any COCO-17 tool)
COCO-17 pose CSVs  (frame, x, y, [z], part_idx, [confidence])
  ↓  kinescope process
Kinematic feature matrix  (position, velocity, acceleration, joint angles, L/R correlations)
  ↓  kinescope predict
Score predictions  (binary or multi-class; logreg / xgboost / random_forest / vit)
```

---

## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/quietscientist/score_prediction_from_video
cd score_prediction_from_video

# Create venv
uv venv .venv
source .venv/bin/activate

uv pip install -e .

# Optional: ViT timeseries model
uv pip install -e ".[vit]"
```

---

## Quick Start

### Step 1 — Run pose estimation

Use [sam3d-video-pose](https://github.com/quietscientist/sam3d-video-pose) or any other tool that outputs COCO-17 keypoints:

```bash
# sam3d-video-pose produces: frame, x, y, z, part_idx
python process_video.py --video my_video.mp4 --prompt "a person"
```

Or convert from another format:

```bash
kinescope convert --input openpose_output.json --format openpose --output my_video_coco.csv
```

### Step 2 — Extract kinematic features

Prepare a video info CSV (`video_info.csv`) with columns: `video`, `fps`, `width`, `height`.

```bash
kinescope process \
  --input ./pose_csvs \
  --output ./pipeline_output \
  --dataset my_study \
  --vid-info ./video_info.csv
# → pipeline_output/my_study_features/features_total_consolidated.csv
```

### Step 3 — Predict scores

Prepare a scores CSV with columns: `video` (or `subject`/`id`) and `score`.

```bash
kinescope predict \
  --features ./pipeline_output/my_study_features/features_total_consolidated.csv \
  --scores ./my_scores.csv \
  --output ./results \
  --model xgboost
```

---

## Python API

```python
import kinescope

# Read any COCO-17 pose CSV (sam3d output)
df = kinescope.read_coco_csv("my_video_pose.csv", video_name="subj001", fps=30)

# Convert from other formats
df = kinescope.convert_to_coco("openpose_output.json", fmt="openpose")
df = kinescope.convert_to_coco("mediapipe_output.csv", fmt="mediapipe")

# Run the full feature extraction pipeline
pipeline = kinescope.PoseProcessingPipeline(
    dataset="my_study",
    pose_dir="./pose_csvs",
    vid_info_csv="./video_info.csv",
    output_path="./output",
)
pipeline.run()

# Train and evaluate a prediction model
from kinescope.prediction import train_and_evaluate, load_features_and_labels
X, y = load_features_and_labels("features.csv", "scores.csv")
results = train_and_evaluate(X, y, model_name="xgboost", output_dir="./results")
print(results["metrics"])
```

---

## Supported Pose Formats

`kinescope convert` accepts any of:

| Format | Source |
|--------|--------|
| `coco` | sam3d-video-pose, MMPose, any COCO-17 CSV |
| `openpose` | OpenPose JSON (body_18) |
| `mediapipe` | MediaPipe Pose (BlazePose 33) CSV |
| `kinect_v2` | KinectV2 25-joint CSV |
| `smpl` | SMPL 24-joint position CSV |

All formats are converted to the canonical COCO-17 DataFrame: `frame, bp, x, y, [z], confidence`.

---

## Kinematic Features

Features are extracted for wrists, ankles, elbows, and knees:

- **Position**: median (x, y), IQR (x, y)
- **Velocity**: median |velocity|, IQR velocity (x, y)
- **Acceleration**: IQR acceleration (x, y)
- **Complexity**: positional entropy
- **Joint angles**: mean, std, entropy, median angular velocity, IQR angular velocity, IQR angular acceleration
- **Left-right coordination**: Pearson correlation between left and right limb trajectories

Both total (whole-video) and windowed (rolling) features are computed.

---

## Prediction Models

| Model | Flag | Notes |
|-------|------|-------|
| Logistic Regression | `logreg` | Fast baseline; L2 regularization |
| XGBoost | `xgboost` | Default; tabular features |
| Random Forest | `random_forest` | Built-in feature importance |
| Pose ViT | `vit` | Small ViT on pose timeseries; requires `[vit]` extra |

All models use `GridSearchCV` + `StratifiedKFold`. Outputs: ROC curve, PR curve, confusion matrix, feature importance, pickled best model.

---

## Package Structure

```
src/kinescope/
├── pose/           # COCO-17 I/O and format converters
├── kinematics/     # Dynamics, joint angles, feature extraction
├── processing/     # Smoothing, interpolation, skeleton normalization
├── pipeline/       # PoseProcessingPipeline (end-to-end feature extraction)
├── prediction/     # Models, GridSearchCV training, evaluation plots
└── skeleton.py     # COCO-17 constants and joint definitions
```

---

## Citation

Feature computation adapted from [Chambers et al. 2020](https://github.com/cchamber/Infant_movement_assessment).

If you use this code, please cite:

```
DOI: 10.5281/zenodo.14042732
```
