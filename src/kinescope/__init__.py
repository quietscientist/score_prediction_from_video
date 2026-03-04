"""
kinescope: Predict clinical and sports scores from human movement video.

Canonical pose format: COCO-17 keypoints.
Supported pose sources: sam3d-video-pose, OpenPose, MediaPipe, KinectV2, SMPL.
"""

from kinescope.pipeline.pose_pipeline import PoseProcessingPipeline
from kinescope.prediction.train import train_and_evaluate
from kinescope.pose.io import read_coco_csv
from kinescope.pose.converter import convert_to_coco, SUPPORTED_FORMATS

__version__ = "0.1.0"
__all__ = [
    "PoseProcessingPipeline",
    "train_and_evaluate",
    "read_coco_csv",
    "convert_to_coco",
    "SUPPORTED_FORMATS",
]
