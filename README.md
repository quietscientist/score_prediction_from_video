# kinescope

**Predict clinical and sports scores from human movement video using kinematics.**

*Powered by Pose-JEPA — self-supervised masked embedding pretraining for skeleton sequences.*

`kinescope` is an open-source Python package for extracting kinematic features from pose estimation data and predicting scores (clinical assessments, sports performance, etc.) using those features. For small labeled datasets (N=20–100), a Vision Transformer pretrained via **Pose-JEPA** (a skeleton adaptation of V-JEPA) provides strong representations before any labels are seen.

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
Score predictions  (regression or classification; logreg / xgboost / random_forest / vit)
```

Two prediction pathways are supported:
- **Kinematic features** — hand-crafted features (velocity, angles, coordination) fed to XGBoost / logistic regression / random forest. Works with N as small as ~20 subjects, but performance follows power law scaling.
- **Pose ViT** — small vision transformer operating directly on COCO-17 timeseries, pretrained via Pose-JEPA (self-supervised masked embedding prediction) on large unlabeled motion datasets, then fine-tuned on labeled data.

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

# Optional: pretraining on large unlabeled datasets
uv pip install -e ".[pretrain]"
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

### (Optional) Step 4 — Pretrain a Pose ViT on unlabeled data

Self-supervised pretraining via Pose-JEPA before fine-tuning on small labeled datasets:

```bash
export KINESCOPE_DATA_DIR=/path/to/pretraining/datasets
kinescope pretrain \
  --datasets amass ntu120 humoto \
  --output ./pretrain_ckpt \
  --epochs 200 \
  --embed-dim 256 \
  --tpc-weight 0.1 \
  --invariant-weight 0.1 \
  --long-horizon-weight 0.05 \
  --long-horizon-segments 4 \
  --artifacts-dir ./artifacts
```

Then use the pretrained weights for fine-tuning:

```bash
kinescope predict \
  --features ./pose_csvs \
  --scores ./my_scores.csv \
  --model vit \
  --pretrained-weights ./pretrain_ckpt/best.pt \
  --output ./results
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

# Train and evaluate a prediction model (kinematic features)
from kinescope.prediction import train_and_evaluate, load_features_and_labels
X, y = load_features_and_labels("features.csv", "scores.csv")
results = train_and_evaluate(X, y, model_name="xgboost", output_dir="./results")
print(results["metrics"])

# Load a labeled clinical dataset (e.g. UDysRS)
from kinescope.linearprobe.udysrs_loader import load_udysrs
data = load_udysrs("/path/to/UDysRS_UPDRS_Export")
# data["arrays"]  — list of (T, 17, 2) COCO-17 clips
# data["scores"]  — z-scored UDysRS totals per task
# data["task"]    — "drinking" | "communication" | "la"
```

The UDysRS dataset (Parkinson's vision-based pose estimation) can be downloaded from Kaggle:
[limi44/parkinsons-visionbased-pose-estimation-dataset](https://www.kaggle.com/datasets/limi44/parkinsons-visionbased-pose-estimation-dataset)
(CC BY 4.0 — Li et al., *J NeuroEng Rehabil* 2018)

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
| Pose ViT | `vit` | Transformer on COCO-17 timeseries; requires `[vit]` extra |

All models use `GridSearchCV` + `StratifiedKFold`. Outputs: ROC curve, PR curve, confusion matrix, feature importance, pickled best model.

### Pose ViT pretraining (Pose-JEPA)

The ViT uses self-supervised pretraining via **Pose-JEPA** (adapted from V-JEPA): an EMA target encoder predicts latent embeddings of spatiotemporally masked joint tokens. A motion-gated temporal predictive coding (TPC) auxiliary loss provides additional signal during dynamic clips. Optional clip-level kinematic invariant supervision can be enabled with `--invariant-weight` to nudge representations toward symmetry/smoothness/coordination/entropy structure. A coarse long-horizon segment objective (`--long-horizon-weight`) can be enabled to bias representations toward slower-timescale temporal structure.

Supported pretraining datasets:

| Dataset | Format | CLI flag |
|---------|--------|----------|
| [AMASS](https://amass.is.tue.mpg.de/) | SMPL params | `amass` |
| [NTU RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) | KinectV2 skeleton | `ntu120` |
| [HUMOTO](https://huggingface.co/datasets/HUMOTO/HUMOTO) | Mixamo GLB + YAML | `humoto` |
| Own COCO-17 CSVs | native | `coco` |

---

## Package Structure

```
src/kinescope/
├── pose/           # COCO-17 I/O and format converters
├── kinematics/     # Dynamics, joint angles, feature extraction
├── processing/     # Smoothing, interpolation, skeleton normalization
├── pipeline/       # PoseProcessingPipeline (end-to-end feature extraction)
├── prediction/     # Models, GridSearchCV training, evaluation plots
│   └── _vit.py     # PoseViT, PoseJEPA, PoseViTClassifier
├── pretrain/       # Pose-JEPA pretraining: data loaders, training loop, visualizations
├── linearprobe/    # Labeled clinical dataset loaders (UDysRS, ...)
└── skeleton.py     # COCO-17 constants and joint definitions
```

---

## Citation

Feature computation adapted from [Chambers et al. 2020](https://github.com/cchamber/Infant_movement_assessment).

If you use this code, please cite:

```
DOI: 10.5281/zenodo.14042732
```
