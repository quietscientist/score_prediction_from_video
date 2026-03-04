"""
PoseProcessingPipeline: end-to-end kinematic feature extraction from COCO-17 pose CSVs.

Accepts output from sam3d-video-pose or any other COCO-17 CSV source.
"""

import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from kinescope.pose.io import read_coco_csv, add_neck
from kinescope.processing.smoothing import interpolate_df, smooth
from kinescope.processing.normalization import normalise_skeletons
from kinescope.kinematics.dynamics import get_dynamics_xy, get_dynamics_angle
from kinescope.kinematics.angles import get_joint_angles
from kinescope.kinematics.features import (
    xy_features, angle_features,
    rolling_xy_features, rolling_angle_features,
    rolling_corr_lr, corr_lr,
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Module-level worker functions (required for multiprocessing pickling)
# ──────────────────────────────────────────────────────────────────────────────

def _process_pose_file(args):
    """Worker: read one COCO-17 CSV → canonical per-keypoint CSV."""
    file_number, file_path, save_path, vid_info, overwrite = args
    file_path = Path(file_path)
    fname = file_path.stem
    out_csv = Path(save_path) / f"{fname}.csv"

    if not overwrite and out_csv.exists():
        return fname

    try:
        info = vid_info[vid_info["video"] == fname]
        if info.empty:
            return None

        fps = float(info["fps"].values[0])
        pixel_x = int(info["width"].values[0])
        pixel_y = int(info["height"].values[0])

        df = read_coco_csv(file_path, video_name=fname, fps=fps)
        df["video_number"] = file_number
        df["pixel_x"] = pixel_x
        df["pixel_y"] = pixel_y
        df["time"] = df["frame"] / fps
        df["delta_t"] = 1 / fps

        if df.empty:
            return None

        df.to_csv(out_csv, index=False)
        return fname

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None


def _smooth_pose_file(args):
    """Worker: interpolate + smooth + normalize one pose CSV."""
    file_path, output_path = args
    file_path = Path(file_path)

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None

        video_name = df["video"].iloc[0]

        df = add_neck(df)

        df = df.groupby(["video", "bp"]).apply(interpolate_df).reset_index(drop=True)

        for coord in ["x", "y"]:
            df = df.groupby(["video", "bp"]).apply(
                lambda g: smooth(g, coord, 0.5, 0.5)
            ).reset_index(drop=True)

        df = normalise_skeletons(df)

        out_path = Path(output_path) / f"{video_name}_smooth_norm.csv"
        df.to_csv(out_path, index=False)
        return video_name

    except Exception as e:
        print(f"Error smoothing {file_path.name}: {e}")
        return None


def _compute_kinematics_file(args):
    """Worker: compute XY dynamics and joint angles for one smooth CSV."""
    file_path, output_path, delta_window = args
    file_path = Path(file_path)
    output_path = Path(output_path)

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None

        video_name = file_path.stem.split("_")[0]

        xdf = get_dynamics_xy(df, delta_window)
        xdf.to_csv(output_path / "xdf" / f"{file_path.stem}_xy.csv", index=False)

        df_with_neck = df if "neck" in df["bp"].values else add_neck(df)
        adf = get_joint_angles(df_with_neck)
        adf = get_dynamics_angle(adf, delta_window)
        adf.to_csv(output_path / "adf" / f"{file_path.stem}_ang.csv", index=False)

        return video_name

    except Exception as e:
        print(f"Error computing kinematics for {file_path.name}: {e}")
        return None


def _extract_xy_features_file(args):
    """Worker: extract XY statistical features from one xdf CSV."""
    file_path, output_path = args
    file_path = Path(file_path)
    output_path = Path(output_path)

    try:
        xdf = pd.read_csv(file_path)
        video_id = xdf["video"].iloc[0]

        limb_bps = ["left_ankle", "right_ankle", "left_wrist", "right_wrist"]
        filtered = xdf[xdf["bp"].isin(limb_bps)]

        # Window features
        feature_xy = xdf.groupby(["bp", "video"]).apply(
            lambda g: rolling_xy_features(g, window_size=60)
        ).reset_index(drop=True)
        feature_xy = pd.pivot_table(feature_xy, index=["video", "frame"], columns="bp")
        feature_xy.columns = [f"{c[0]}_{c[1]}" for c in feature_xy.columns]
        feature_xy = feature_xy.reset_index()

        xdf["dist"] = np.sqrt(xdf["x"] ** 2 + xdf["y"] ** 2)
        corr_j = xdf[xdf["bp"].isin(limb_bps)].groupby(["video", "part"]).apply(
            lambda g: rolling_corr_lr(g, var="dist")
        ).reset_index()
        if not corr_j.empty and len(corr_j.columns) > 3:
            val_cols = [c for c in corr_j.columns
                        if c not in ["video", "part", "level_2", "R", "L"]]
            if val_cols:
                corr_j = corr_j.drop(
                    columns=[c for c in ["level_2", "R", "L"] if c in corr_j.columns])
                corr_j["part"] = "lrCorr_x_" + corr_j["part"]
                corr_j = pd.pivot_table(
                    corr_j, index=["video", "frame"], columns="part", values=val_cols[0]
                ).reset_index()
                feature_xy = pd.merge(feature_xy, corr_j, on=["video", "frame"], how="outer")

        feature_xy.to_csv(
            output_path / "xy_features/windows" / f"{video_id}_features_windows_xy.csv",
            index=False,
        )

        # Total features
        feature_total = filtered.groupby(["bp", "video"]).apply(
            xy_features
        ).reset_index(drop=True)
        feature_total = pd.pivot_table(feature_total, index="video", columns="bp")
        feature_total.columns = [f"{c[0]}_{c[1]}" for c in feature_total.columns]
        feature_total = feature_total.reset_index()

        corr_t = xdf.groupby(["video", "part"]).apply(
            lambda g: corr_lr(g, "dist")
        ).reset_index()
        if not corr_t.empty and len(corr_t.columns) > 2:
            val_cols = [c for c in corr_t.columns if c not in ["video", "part"]]
            if val_cols:
                corr_t["part"] = "lrCorr_x_" + corr_t["part"]
                corr_t = pd.pivot_table(
                    corr_t, index="video", columns="part", values=val_cols[0]
                ).reset_index()
                feature_total = pd.merge(feature_total, corr_t, on="video", how="outer")

        feature_total.to_csv(
            output_path / "xy_features/total" / f"{video_id}_features_total_xy.csv",
            index=False,
        )
        return video_id

    except Exception as e:
        import traceback
        print(f"Error extracting XY features from {file_path.name}: {e}")
        print(traceback.format_exc())
        return None


def _extract_angle_features_file(args):
    """Worker: extract angle statistical features from one adf CSV."""
    file_path, output_path = args
    file_path = Path(file_path)
    output_path = Path(output_path)

    try:
        adf = pd.read_csv(file_path)
        video_id = adf["video"].iloc[0]
        window_size = 2 * int(adf["fps"].iloc[0])

        feature_angle = adf.groupby(["bp", "video"]).apply(
            lambda g: rolling_angle_features(g, window_size=window_size)
        ).reset_index(drop=True)
        feature_angle = pd.pivot_table(feature_angle, index=["video", "frame"], columns="bp")
        feature_angle.columns = [f"{c[0]}_{c[1]}" for c in feature_angle.columns]
        feature_angle = feature_angle.reset_index()

        corr_j = adf.groupby(["video", "part"]).apply(
            lambda g: rolling_corr_lr(g, window_size=window_size, min_periods=1, var="angle")
        ).reset_index()
        if not corr_j.empty and len(corr_j.columns) > 3:
            val_cols = [c for c in corr_j.columns
                        if c not in ["video", "part", "level_2", "R", "L"]]
            if val_cols:
                corr_j = corr_j.drop(
                    columns=[c for c in ["level_2", "R", "L"] if c in corr_j.columns])
                corr_j["part"] = "lrCorr_angle_" + corr_j["part"]
                corr_j = pd.pivot_table(
                    corr_j, index=["video", "frame"], columns="part", values=val_cols[0]
                ).reset_index()
                feature_angle = pd.merge(feature_angle, corr_j,
                                         on=["video", "frame"], how="outer")

        feature_angle.to_csv(
            output_path / "angle_features/windows" / f"{video_id}_features_windows_angle.csv",
            index=False,
        )

        feature_total = adf.groupby(["bp", "video"]).apply(
            angle_features
        ).reset_index(drop=True)
        feature_total = pd.pivot_table(feature_total, index="video", columns="bp")
        feature_total.columns = [f"{c[0]}_{c[1]}" for c in feature_total.columns]
        feature_total = feature_total.reset_index()

        corr_t = adf.groupby(["video", "part"]).apply(
            lambda g: corr_lr(g, "angle")
        ).reset_index()
        if not corr_t.empty and len(corr_t.columns) > 2:
            val_cols = [c for c in corr_t.columns if c not in ["video", "part"]]
            if val_cols:
                corr_t["part"] = "lrCorr_angle_" + corr_t["part"]
                corr_t = pd.pivot_table(
                    corr_t, index="video", columns="part", values=val_cols[0]
                ).reset_index()
                feature_total = pd.merge(feature_total, corr_t, on="video", how="outer")

        feature_total.to_csv(
            output_path / "angle_features/total" / f"{video_id}_features_total_angle.csv",
            index=False,
        )
        return video_id

    except Exception as e:
        import traceback
        print(f"Error extracting angle features from {file_path.name}: {e}")
        print(traceback.format_exc())
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class PoseProcessingPipeline:
    """
    End-to-end pipeline: COCO-17 pose CSVs → consolidated kinematic feature matrix.

    Stages:
      1. Read COCO-17 CSVs + attach video metadata
      2. Interpolate, smooth, normalize skeletons
      3. Compute XY dynamics and joint angles
      4. Extract statistical features (total + windowed)
      5. Consolidate into a single feature matrix

    Usage
    -----
    pipeline = PoseProcessingPipeline(
        dataset="my_study",
        pose_dir="./poses",
        vid_info_csv="./my_study_video_info.csv",
    )
    pipeline.run()
    """

    def __init__(
        self,
        dataset: str,
        pose_dir: str,
        vid_info_csv: str,
        output_path: str = "./pipeline_output",
        n_workers: int = 8,
        overwrite: bool = True,
    ):
        self.dataset = dataset
        self.pose_dir = Path(pose_dir)
        self.output_path = Path(output_path) / f"{dataset}_features"
        self.vid_info_csv = Path(vid_info_csv)
        self.n_workers = n_workers
        self.overwrite = overwrite

        self.log_path = self.output_path / "logs"
        self._data_loss: Dict[str, dict] = {}
        self._log_file: Path = None  # set in setup_directories

    def setup_directories(self):
        subdirs = [
            "pose_estimates", "smooth", "xdf", "adf",
            "xy_features/total", "xy_features/windows",
            "angle_features/total", "angle_features/windows",
            "logs",
        ]
        for s in subdirs:
            (self.output_path / s).mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

    def load_video_info(self) -> pd.DataFrame:
        vid_info = pd.read_csv(self.vid_info_csv)
        self.log(f"Loaded video info: {len(vid_info)} videos")
        return vid_info

    def log(self, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(f"[{ts}] {message}\n")
        print(message)

    def track_data_loss(self, stage: str, inputs: list, outputs: list):
        in_set = set(str(x) for x in inputs)
        out_set = set(str(x) for x in outputs)
        lost = in_set - out_set
        loss_pct = len(lost) / len(in_set) * 100 if in_set else 0
        self._data_loss[stage] = {
            "input": len(in_set), "output": len(out_set),
            "lost": len(lost), "lost_items": list(lost),
        }
        self.log(
            f"[{stage}] input={len(in_set)} output={len(out_set)} "
            f"lost={len(lost)} ({loss_pct:.1f}%)"
        )

    def save_data_loss_report(self):
        rows = [{"stage": k, **v} for k, v in self._data_loss.items()]
        df = pd.DataFrame(rows)
        report_path = self.log_path / "data_loss_report.csv"
        df.to_csv(report_path, index=False)
        self.log(f"Data loss report → {report_path}")

    def process_pose_annotations(self, vid_info: pd.DataFrame):
        self.log("Stage 1: Processing pose CSVs")
        pose_files = sorted(self.pose_dir.glob("*.csv"))
        save_path = self.output_path / "pose_estimates"
        args = [
            (i, str(f), str(save_path), vid_info, self.overwrite)
            for i, f in enumerate(pose_files)
        ]
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(pool.imap(_process_pose_file, args),
                                total=len(pose_files), desc="Reading poses"))
        processed = [r for r in results if r is not None]
        self.track_data_loss("Pose CSVs", [f.stem for f in pose_files], processed)

    def smooth_and_normalize(self):
        self.log("Stage 2: Smoothing and normalization")
        pose_files = list((self.output_path / "pose_estimates").glob("*.csv"))
        args = [(str(f), str(self.output_path / "smooth")) for f in pose_files]
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(pool.imap(_smooth_pose_file, args),
                                total=len(pose_files), desc="Smoothing"))
        processed = [r for r in results if r is not None]
        self.track_data_loss("Smoothing", [f.stem for f in pose_files], processed)

    def compute_kinematics(self):
        self.log("Stage 3: Computing kinematics")
        smooth_files = list((self.output_path / "smooth").glob("*.csv"))
        args = [(str(f), str(self.output_path), 0.25) for f in smooth_files]
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(pool.imap(_compute_kinematics_file, args),
                                total=len(smooth_files), desc="Kinematics"))
        processed = [r for r in results if r is not None]
        self.track_data_loss("Kinematics",
                             [f.stem.split("_")[0] for f in smooth_files], processed)

    def extract_features(self):
        self.log("Stage 4: Extracting features")
        xdf_files = list((self.output_path / "xdf").glob("*.csv"))
        args = [(str(f), str(self.output_path)) for f in xdf_files]
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(pool.imap(_extract_xy_features_file, args),
                                total=len(xdf_files), desc="XY features"))
        processed = [r for r in results if r is not None]
        self.track_data_loss("XY Features",
                             [f.stem.split("_")[0] for f in xdf_files], processed)

        adf_files = list((self.output_path / "adf").glob("*.csv"))
        args = [(str(f), str(self.output_path)) for f in adf_files]
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(pool.imap(_extract_angle_features_file, args),
                                total=len(adf_files), desc="Angle features"))
        processed = [r for r in results if r is not None]
        self.track_data_loss("Angle Features",
                             [f.stem.split("_")[0] for f in adf_files], processed)

    def consolidate_features(self):
        self.log("Stage 5: Consolidating features")
        xy_files = list((self.output_path / "xy_features/total").glob("*.csv"))
        angle_files = list((self.output_path / "angle_features/total").glob("*.csv"))

        if not xy_files or not angle_files:
            self.log("No feature files found — skipping consolidation.")
            return

        features_xy = pd.concat([pd.read_csv(f) for f in xy_files], ignore_index=True)
        features_angle = pd.concat([pd.read_csv(f) for f in angle_files], ignore_index=True)
        features_total = pd.merge(features_xy, features_angle, on="video", how="inner")

        keywords = ["wrist", "ankle", "elbow", "knee"]
        limb_cols = [c for c in features_total.columns
                     if any(k in c.lower() for k in keywords)]
        features_total = features_total[["video"] + limb_cols]

        out_path = self.output_path / "features_total_consolidated.csv"
        features_total.to_csv(out_path, index=False)
        self.log(f"Consolidated features: {features_total.shape} → {out_path}")

        self.track_data_loss("Consolidation",
                             features_xy["video"].unique(), features_total["video"].unique())

    def run(self):
        self.log("=" * 60)
        self.log(f"kinescope PoseProcessingPipeline — dataset: {self.dataset}")
        self.log("=" * 60)

        self.setup_directories()
        vid_info = self.load_video_info()

        self.process_pose_annotations(vid_info)
        self.smooth_and_normalize()
        self.compute_kinematics()
        self.extract_features()
        self.consolidate_features()
        self.save_data_loss_report()

        self.log("=" * 60)
        self.log(f"Pipeline complete → {self.output_path}")
        self.log("=" * 60)
