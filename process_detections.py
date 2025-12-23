#!/usr/bin/env python3
"""
Infant Movement Pose Processing Pipeline
Processes pose estimation JSON files to extract kinematic features for GMA scoring.
"""

import os
import sys
import json
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from multiprocessing import Pool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import custom modules (assumed to be in ./utils)
sys.path.append('./utils')
from processing import *
from kinematics import *
from skeleton import *

warnings.filterwarnings("ignore")


# Global worker functions for multiprocessing (must be at module level)
def _process_json_file(args):
    """Worker function for processing JSON pose annotations."""
    file_number, file, json_path, save_path, vid_info, kp_mapping, overwrite = args
    file_path = json_path / file
    fname = file.stem
    
    if not overwrite and (save_path / f'{fname}.csv').exists():
        return fname
    
    try:
        with open(file_path, 'r') as f:
            frames = json.load(f)
            info = vid_info[vid_info['video'] == fname]
            
            if info.empty:
                return None
            
            fps = info['fps'].values[0]
            pixel_x = info['width'].values[0]
            pixel_y = info['height'].values[0]
            
            data = []
            for frame in frames:
                frame_id = frame['frame_id']
                if 'instances' not in frame or not frame['instances']:
                    continue
                
                # Get best instance (highest confidence)
                instance = max(frame['instances'], 
                             key=lambda x: sum(x['keypoint_scores']))
                
                keypoints = instance['keypoints']
                confidence = instance['keypoint_scores']
                keypoints, confidence = convert_coco_to_openpose(
                    keypoints, confidence
                )
                
                for part_idx, (x, y) in enumerate(keypoints):
                    data.append([
                        file_number, fname, kp_mapping[part_idx],
                        frame_id, x, y, confidence[part_idx],
                        fps, pixel_x, pixel_y, frame_id / fps, part_idx
                    ])
        
        if data:
            columns = ['video_number', 'video', 'bp', 'frame', 'x', 'y', 'c',
                     'fps', 'pixel_x', 'pixel_y', 'time', 'part_idx']
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(save_path / f'{fname}.csv', index=False)
            return fname
        
        return None
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None


def _smooth_pose_file(args):
    """Worker function for smoothing pose data."""
    file_path, output_path = args
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        # Parse video metadata
        infant = df['video'].iloc[0]
        session = 0
        age = '3Month'
        
        # Interpolate missing values
        df = df.groupby(['video', 'bp']).apply(
            interpolate_df
        ).reset_index(drop=True)
        
        # Apply smoothing
        median_window = 0.5
        mean_window = 0.5
        for coord in ['x', 'y']:
            df = df.groupby(['video', 'bp']).apply(
                lambda x: smooth(x, coord, median_window, mean_window)
            ).reset_index(drop=True)
        
        # Normalize
        df = normalise_skeletons(df)
        
        # Save smoothed data
        out_path = output_path / f'{infant}_{session}_{age}_smooth_norm.csv'
        df.to_csv(out_path, index=False)
        
        return infant
        
    except Exception as e:
        print(f"Error smoothing {file_path.name}: {e}")
        return None


def _compute_kinematics_file(args):
    """Worker function for computing kinematics."""
    file_path, output_path, delta_window = args
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        infant = file_path.stem.split('_')[0]
        session = file_path.stem.split('_')[1]
        age = file_path.stem.split('_')[2]
        
        # XY dynamics
        xdf = get_dynamics_xy(df, delta_window)
        xdf.to_csv(
            output_path / 'xdf' / f'{infant}_{session}_{age}_smooth_norm_xy.csv',
            index=False
        )
        
        # Joint angles
        adf = get_joint_angles(df)
        adf = get_dynamics_angle(adf, delta_window)
        adf.to_csv(
            output_path / 'adf' / f'{infant}_{session}_{age}_smooth_norm_ang.csv',
            index=False
        )
        
        return infant
        
    except Exception as e:
        print(f"Error computing kinematics for {file_path.name}: {e}")
        return None


def _extract_xy_features_file(args):
    """Worker function for extracting XY features."""
    file_path, output_path = args
    
    try:
        xdf = pd.read_csv(file_path)
        video_number = xdf['video'].iloc[0]
        
        bps = ['LAnkle', 'RAnkle', 'LWrist', 'RWrist']
        filtered_xdf = xdf[xdf['bp'].isin(bps)]
        
        # Window features
        feature_xy = xdf.groupby(['bp', 'video']).apply(
            lambda g: rolling_xy_features(g, window_size=60)
        ).reset_index(drop=True)
        
        feature_xy = pd.pivot_table(
            feature_xy, index=['video', 'frame'], columns='bp'
        )
        feature_xy.columns = [f'{c[0]}_{c[1]}' for c in feature_xy.columns]
        feature_xy = feature_xy.reset_index()
        
        # Add symmetry correlations
        xdf['dist'] = np.sqrt(xdf['x']**2 + xdf['y']**2)
        corr_joint = xdf[xdf['bp'].isin(bps)].groupby(['video', 'part']).apply(
            lambda x: rolling_corr_lr(x, var='dist')
        ).reset_index()
        
        if not corr_joint.empty and len(corr_joint.columns) > 3:
            corr_joint['part'] = 'lrCorr_x_' + corr_joint['part']
            # Get the column with correlation values (should be the last numeric column)
            value_col = [c for c in corr_joint.columns if c not in ['video', 'part', 'level_2', 'R', 'L']]
            if value_col:
                corr_joint = corr_joint.drop(columns=[c for c in ['level_2', 'R', 'L'] if c in corr_joint.columns])
                corr_joint = pd.pivot_table(
                    corr_joint, index=['video', 'frame'], columns='part', values=value_col[0]
                )
                corr_joint = corr_joint.reset_index()
                feature_xy = pd.merge(feature_xy, corr_joint, on=['video', 'frame'], how='outer')
        
        feature_xy.to_csv(
            output_path / 'xy_features/windows' / f'{video_number}_features_windows_xy.csv',
            index=False
        )
        
        # Total features (averaged)
        feature_total = filtered_xdf.groupby(['bp', 'video']).apply(
            xy_features
        ).reset_index(drop=True)
        
        feature_total = pd.pivot_table(feature_total, index='video', columns='bp')
        feature_total.columns = [f'{c[0]}_{c[1]}' for c in feature_total.columns]
        feature_total = feature_total.reset_index()
        
        corr_total = xdf.groupby(['video', 'part']).apply(
            lambda x: corr_lr(x, 'dist')
        ).reset_index()
        
        if not corr_total.empty and len(corr_total.columns) > 2:
            corr_total['part'] = 'lrCorr_x_' + corr_total['part']
            # Get the column with correlation values
            value_col = [c for c in corr_total.columns if c not in ['video', 'part']]
            if value_col:
                corr_total = pd.pivot_table(corr_total, index='video', columns='part', values=value_col[0])
                corr_total = corr_total.reset_index()
                feature_total = pd.merge(feature_total, corr_total, on='video', how='outer')
        
        feature_total.to_csv(
            output_path / 'xy_features/total' / f'{video_number}_features_total_xy.csv',
            index=False
        )
        
        return video_number
        
    except Exception as e:
        import traceback
        print(f"Error extracting XY features from {file_path.name}: {e}")
        print(traceback.format_exc())
        return None


def _extract_angle_features_file(args):
    """Worker function for extracting angle features."""
    file_path, output_path = args
    
    try:
        adf = pd.read_csv(file_path)
        video_number = adf['video'].iloc[0]
        window_size = 2 * int(adf['fps'].iloc[0])
        
        # Window features
        feature_angle = adf.groupby(['bp', 'video']).apply(
            rolling_angle_features, window_size=window_size
        ).reset_index(drop=True)
        
        feature_angle = pd.pivot_table(
            feature_angle, index=['video', 'frame'], columns='bp'
        )
        feature_angle.columns = [f'{c[0]}_{c[1]}' for c in feature_angle.columns]
        feature_angle = feature_angle.reset_index()
        
        # Symmetry correlations
        corr_joint = adf.groupby(['video', 'part']).apply(
            rolling_corr_lr, window_size=window_size, min_periods=1, var='angle'
        ).reset_index()
        
        if not corr_joint.empty and len(corr_joint.columns) > 3:
            # Get the column with correlation values
            value_col = [c for c in corr_joint.columns if c not in ['video', 'part', 'level_2', 'R', 'L']]
            if value_col:
                corr_joint = corr_joint.drop(columns=[c for c in ['level_2', 'R', 'L'] if c in corr_joint.columns])
                corr_joint['part'] = 'lrCorr_angle_' + corr_joint['part']
                corr_joint = pd.pivot_table(
                    corr_joint, index=['video', 'frame'], columns='part', values=value_col[0]
                )
                corr_joint = corr_joint.reset_index()
                feature_angle = pd.merge(feature_angle, corr_joint, on=['video', 'frame'], how='outer')
        
        feature_angle.to_csv(
            output_path / 'angle_features/windows' / f'{video_number}_features_windows_angle.csv',
            index=False
        )
        
        # Total features
        feature_total = adf.groupby(['bp', 'video']).apply(
            angle_features
        ).reset_index(drop=True)
        
        feature_total = pd.pivot_table(feature_total, index='video', columns='bp')
        feature_total.columns = [f'{c[0]}_{c[1]}' for c in feature_total.columns]
        feature_total = feature_total.reset_index()
        
        corr_total = adf.groupby(['video', 'part']).apply(
            lambda x: corr_lr(x, 'angle')
        ).reset_index()
        
        if not corr_total.empty and len(corr_total.columns) > 2:
            corr_total['part'] = 'lrCorr_angle_' + corr_total['part']
            # Get the column with correlation values
            value_col = [c for c in corr_total.columns if c not in ['video', 'part']]
            if value_col:
                corr_total = pd.pivot_table(corr_total, index='video', columns='part', values=value_col[0])
                corr_total = corr_total.reset_index()
                feature_total = pd.merge(feature_total, corr_total, on='video', how='outer')
        
        feature_total.to_csv(
            output_path / 'angle_features/total' / f'{video_number}_features_total_angle.csv',
            index=False
        )
        
        return video_number
        
    except Exception as e:
        import traceback
        print(f"Error extracting angle features from {file_path.name}: {e}")
        print(traceback.format_exc())
        return None


class PoseProcessingPipeline:
    """Main pipeline for processing pose estimation data."""
    
    def __init__(self, dataset: str, base_path: str = './data', 
                 n_workers: int = 20, overwrite: bool = True):
        """
        Initialize pipeline.
        
        Args:
            dataset: Dataset name (e.g., 'PANDA2')
            base_path: Root directory for data
            n_workers: Number of parallel workers
            overwrite: Whether to overwrite existing files
        """
        self.dataset = dataset
        self.base_path = Path(base_path)
        self.n_workers = n_workers
        self.overwrite = overwrite
        
        # Setup paths
        self.json_path = self.base_path / 'annotations'
        self.output_path = Path(f'./pose_estimates/{dataset}_pose_estimates')
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_path = self.output_path / 'logs'
        self.log_path.mkdir(exist_ok=True)
        self.log_file = self.log_path / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'
        
        # Data loss tracking
        self.data_loss_log = {
            'stage': [],
            'input_count': [],
            'output_count': [],
            'lost_count': [],
            'lost_items': [],
            'timestamp': []
        }
        
        # Keypoint mapping (COCO to OpenPose)
        self.kp_mapping = {
            0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
            5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'RHip', 9: 'RKnee',
            10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'REye',
            15: 'LEye', 16: 'REar', 17: 'LEar'
        }
        
        # Feature selection mapping
        self.feature_mapping = self._get_feature_mapping()
        
        # Pipeline state tracking
        self.pipeline_state = self._check_pipeline_state()
        
        self.log("Pipeline initialized")
        self.log(f"Dataset: {dataset}")
        self.log(f"Output path: {self.output_path}")
        self.log(f"Overwrite mode: {overwrite}")
    
    def _check_pipeline_state(self) -> dict:
        """Check which stages have been completed."""
        state = {
            'json_processing_done': False,
            'smoothing_done': False,
            'kinematics_done': False,
            'xy_features_done': False,
            'angle_features_done': False,
            'consolidation_done': False,
            'json_count': 0,
            'smooth_count': 0,
            'xdf_count': 0,
            'adf_count': 0,
            'xy_total_count': 0,
            'xy_window_count': 0,
            'angle_total_count': 0,
            'angle_window_count': 0,
        }
        
        # Check JSON processing
        pose_estimates_path = self.output_path / 'pose_estimates'
        if pose_estimates_path.exists():
            csv_files = list(pose_estimates_path.glob('*.csv'))
            state['json_count'] = len(csv_files)
            state['json_processing_done'] = state['json_count'] > 0
        
        # Check smoothing
        smooth_path = self.output_path / 'smooth'
        if smooth_path.exists():
            smooth_files = list(smooth_path.glob('*_smooth_norm.csv'))
            state['smooth_count'] = len(smooth_files)
            state['smoothing_done'] = state['smooth_count'] > 0
        
        # Check kinematics
        xdf_path = self.output_path / 'xdf'
        adf_path = self.output_path / 'adf'
        if xdf_path.exists():
            xdf_files = list(xdf_path.glob('*_xy.csv'))
            state['xdf_count'] = len(xdf_files)
        if adf_path.exists():
            adf_files = list(adf_path.glob('*_ang.csv'))
            state['adf_count'] = len(adf_files)
        state['kinematics_done'] = state['xdf_count'] > 0 and state['adf_count'] > 0
        
        # Check XY features
        xy_total_path = self.output_path / 'xy_features/total'
        xy_window_path = self.output_path / 'xy_features/windows'
        if xy_total_path.exists():
            xy_total_files = list(xy_total_path.glob('*.csv'))
            state['xy_total_count'] = len(xy_total_files)
        if xy_window_path.exists():
            xy_window_files = list(xy_window_path.glob('*.csv'))
            state['xy_window_count'] = len(xy_window_files)
        state['xy_features_done'] = state['xy_total_count'] > 0
        
        # Check angle features
        angle_total_path = self.output_path / 'angle_features/total'
        angle_window_path = self.output_path / 'angle_features/windows'
        if angle_total_path.exists():
            angle_total_files = list(angle_total_path.glob('*.csv'))
            state['angle_total_count'] = len(angle_total_files)
        if angle_window_path.exists():
            angle_window_files = list(angle_window_path.glob('*.csv'))
            state['angle_window_count'] = len(angle_window_files)
        state['angle_features_done'] = state['angle_total_count'] > 0
        
        # Check consolidation
        consolidated_path = self.output_path / 'features_total_consolidated.csv'
        state['consolidation_done'] = consolidated_path.exists()
        
        return state
    
    def _log_pipeline_state(self):
        """Log current pipeline state."""
        self.log("\n" + "="*70)
        self.log("PIPELINE STATE CHECK")
        self.log("="*70)
        
        stages = [
            ("JSON Processing", "json_processing_done", "json_count"),
            ("Smoothing", "smoothing_done", "smooth_count"),
            ("Kinematics", "kinematics_done", None),
            ("  - XY Dynamics", None, "xdf_count"),
            ("  - Angle Dynamics", None, "adf_count"),
            ("XY Features", "xy_features_done", None),
            ("  - Total", None, "xy_total_count"),
            ("  - Windows", None, "xy_window_count"),
            ("Angle Features", "angle_features_done", None),
            ("  - Total", None, "angle_total_count"),
            ("  - Windows", None, "angle_window_count"),
            ("Consolidation", "consolidation_done", None),
        ]
        
        for stage_name, done_key, count_key in stages:
            status = ""
            if done_key and self.pipeline_state[done_key]:
                status = "✓ DONE"
            elif done_key:
                status = "✗ PENDING"
            
            count_info = ""
            if count_key and self.pipeline_state[count_key] > 0:
                count_info = f" ({self.pipeline_state[count_key]} files)"
            
            if status or count_info:
                self.log(f"{stage_name:30s} {status:10s} {count_info}")
        
        self.log("="*70 + "\n")
    
    def log(self, message: str):
        """Write to log file with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_msg)
        print(message)
    
    def track_data_loss(self, stage: str, input_items: List, output_items: List):
        """Track data loss between pipeline stages."""
        input_set = set(str(x) for x in input_items)
        output_set = set(str(x) for x in output_items)
        lost = input_set - output_set
        
        self.data_loss_log['stage'].append(stage)
        self.data_loss_log['input_count'].append(len(input_set))
        self.data_loss_log['output_count'].append(len(output_set))
        self.data_loss_log['lost_count'].append(len(lost))
        self.data_loss_log['lost_items'].append(list(lost))
        self.data_loss_log['timestamp'].append(datetime.now())
        
        loss_pct = len(lost) / len(input_set) * 100 if input_set else 0
        self.log(f"Stage: {stage} | Input: {len(input_set)} | "
                f"Output: {len(output_set)} | Lost: {len(lost)} ({loss_pct:.1f}%)")
        
        if lost:
            self.log(f"Lost items: {list(lost)[:10]}{'...' if len(lost) > 10 else ''}")
    
    def save_data_loss_report(self):
        """Save comprehensive data loss report."""
        df = pd.DataFrame(self.data_loss_log)
        report_path = self.log_path / 'data_loss_report.csv'
        df.to_csv(report_path, index=False)
        self.log(f"Data loss report saved to {report_path}")
        
        # Create visualization
        self._plot_data_loss(df)
    
    def _plot_data_loss(self, df: pd.DataFrame):
        """Create data loss visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot absolute counts
        x = range(len(df))
        ax1.plot(x, df['input_count'], 'o-', label='Input', linewidth=2)
        ax1.plot(x, df['output_count'], 'o-', label='Output', linewidth=2)
        ax1.fill_between(x, df['output_count'], df['input_count'], alpha=0.3, color='red')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['stage'], rotation=45, ha='right')
        ax1.set_ylabel('Count')
        ax1.set_title('Data Flow Through Pipeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss percentage
        loss_pct = (df['lost_count'] / df['input_count'] * 100).fillna(0)
        ax2.bar(x, loss_pct, color='red', alpha=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['stage'], rotation=45, ha='right')
        ax2.set_ylabel('Loss (%)')
        ax2.set_title('Data Loss Percentage by Stage')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.log_path / 'data_loss_visualization.png', dpi=150)
        plt.close()
        self.log("Data loss visualization saved")
    
    def _get_feature_mapping(self) -> Dict[str, str]:
        """Define mapping from technical to readable feature names."""
        return {
            "IQR_x_Ankle": "IQR_ankle_pos_x",
            "median_vel_x_Ankle": "Med_ankle_vel_x",
            "IQR_vel_x_Ankle": "IQR_ankle_vel_x",
            "mean_ent_Ankle": "Entropy_ankle_pos",
            "IQR_acc_x_Ankle": "IQR_ankle_accel_x",
            "median_vel_y_Ankle": "Med_ankle_vel_y",
            "IQR_vel_y_Ankle": "IQR_ankle_vel_y",
            "median_vel_Knee": "Med_knee_angle_vel",
            "IQR_vel_Knee": "IQR_knee_angle_vel",
            "stdev_angle_Knee": "Stdev_knee_angle",
            "median_y_Wrist": "Med_wrist_pos_y",
            "IQR_y_Ankle": "IQR_ankle_pos_y",
            "entropy_angle_Knee": "Entropy_knee_angle",
            "IQR_acc_Knee": "IQR_knee_angle_accel",
            "mean_angle_Elbow": "Mean_elbow_angle",
            "IQR_acc_y_Ankle": "IQR_ankle_accel_y",
            "median_x_Ankle": "Med_ankle_pos_x",
            "median_x_Wrist": "Med_wrist_pos_x",
            "median_y_Ankle": "Med_ankle_pos_y",
            "lrCorr_angle_Elbow": "Cross-corr_elbow_angle",
            "lrCorr_x_Wrist": "Cross-corr_wrist_pos",
            "mean_angle_Knee": "Mean_knee_angle",
            "lrCorr_x_Ankle": "Cross-corr_ankle_pos",
            "IQR_x_Wrist": "IQR_wrist_pos_x",
            "lrCorr_angle_Knee": "Cross-corr_knee_angle",
            "IQR_y_Wrist": "IQR_wrist_pos_y",
            "mean_ent_Wrist": "Entropy_wrist_pos",
            "median_vel_x_Wrist": "Med_wrist_vel_x",
            "IQR_vel_x_Wrist": "IQR_wrist_vel_x",
            "IQR_acc_x_Wrist": "IQR_wrist_accel_x",
            "median_vel_y_Wrist": "Med_wrist_vel_y",
            "IQR_vel_y_Wrist": "IQR_wrist_vel_y",
            "IQR_acc_y_Wrist": "IQR_wrist_accel_y",
            "stdev_angle_Elbow": "Stdev_elbow_angle",
            "IQR_acc_Elbow": "IQR_elbow_angle_accel",
            "entropy_angle_Elbow": "Entropy_elbow_angle",
            "IQR_vel_Elbow": "IQR_elbow_angle_vel",
            "median_vel_Elbow": "Med_elbow_angle_vel",
        }
    
    def setup_directories(self):
        """Create necessary directory structure."""
        subdirs = [
            'pose_estimates', 'xdf', 'adf', 'smooth', 'anim',
            'xy_features/total', 'xy_features/windows',
            'angle_features/total', 'angle_features/windows',
            'window_features', 'distributions'
        ]
        
        for subdir in subdirs:
            (self.output_path / subdir).mkdir(parents=True, exist_ok=True)
        
        self.log("Directory structure created")
    
    def load_video_info(self) -> pd.DataFrame:
        """Load and preprocess video metadata."""
        vid_info = pd.read_csv(self.base_path / f'{self.dataset}_video_info.csv')
        
        # Dataset-specific preprocessing
        vid_info = vid_info[~vid_info['video'].str.endswith('_0')].reset_index(drop=True)
        vid_info['ID'] = vid_info['video'].str.split('_').str[0]
        vid_info = vid_info.sort_values('video').drop_duplicates(
            subset=['ID'], keep='last'
        ).reset_index(drop=True)
        vid_info = vid_info.drop(columns=['ID'])
        vid_info['video'] = vid_info['video'].str.split('_').str[0]
        
        self.log(f"Loaded video info: {len(vid_info)} videos")
        return vid_info
    
    def process_json_annotations(self, vid_info: pd.DataFrame):
        """Process JSON pose annotations to CSV format."""
        self.log("Stage 1: Processing JSON annotations")
        
        json_files = [f for f in os.listdir(self.json_path) if f.endswith('.json')]
        save_path = self.output_path / 'pose_estimates'
        
        # Parallel processing
        args = [
            (i, Path(f), self.json_path, save_path, vid_info, self.kp_mapping, self.overwrite)
            for i, f in enumerate(json_files)
        ]
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(pool.imap(_process_json_file, args), 
                              total=len(json_files), desc="Processing JSONs"))
        
        processed = [r for r in results if r is not None]
        input_videos = [Path(f).stem for f in json_files]
        self.track_data_loss("JSON to CSV", input_videos, processed)
    
    def smooth_and_normalize(self):
        """Apply smoothing and normalization to pose estimates."""
        self.log("Stage 2: Smoothing and normalization")
        
        pose_files = list((self.output_path / 'pose_estimates').glob('*.csv'))
        
        # Prepare arguments
        args = [(f, self.output_path / 'smooth') for f in pose_files]
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(_smooth_pose_file, args),
                total=len(pose_files),
                desc="Smoothing"
            ))
        
        processed = [r for r in results if r is not None]
        input_ids = [f.stem for f in pose_files]
        self.track_data_loss("Smoothing", input_ids, processed)
    
    def compute_kinematics(self):
        """Compute kinematic features (XY dynamics and joint angles)."""
        self.log("Stage 3: Computing kinematics")
        
        smooth_files = list((self.output_path / 'smooth').glob('*.csv'))
        delta_window = 0.25
        
        # Prepare arguments
        args = [(f, self.output_path, delta_window) for f in smooth_files]
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_kinematics_file, args),
                total=len(smooth_files),
                desc="Computing kinematics"
            ))
        
        processed = [r for r in results if r is not None]
        input_ids = [f.stem.split('_')[0] for f in smooth_files]
        self.track_data_loss("Kinematics", input_ids, processed)
    
    def extract_features(self):
        """Extract statistical features from kinematic data."""
        self.log("Stage 4: Extracting features")
        
        # XY features
        self._extract_xy_features()
        
        # Angle features
        self._extract_angle_features()
    
    def _extract_xy_features(self):
        """Extract XY coordinate features."""
        xdf_files = list((self.output_path / 'xdf').glob('*.csv'))
        
        # Prepare arguments
        args = [(f, self.output_path) for f in xdf_files]
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(_extract_xy_features_file, args),
                total=len(xdf_files),
                desc="XY features"
            ))
        
        processed = [r for r in results if r is not None]
        input_ids = [f.stem.split('_')[0] for f in xdf_files]
        self.track_data_loss("XY Features", input_ids, processed)
    
    def _extract_angle_features(self):
        """Extract joint angle features."""
        adf_files = list((self.output_path / 'adf').glob('*.csv'))
        
        # Prepare arguments
        args = [(f, self.output_path) for f in adf_files]
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(_extract_angle_features_file, args),
                total=len(adf_files),
                desc="Angle features"
            ))
        
        processed = [r for r in results if r is not None]
        input_ids = [f.stem.split('_')[0] for f in adf_files]
        self.track_data_loss("Angle Features", input_ids, processed)
    
    def consolidate_features(self):
        """Merge and consolidate all features into final matrices."""
        self.log("Stage 5: Consolidating features")
        
        # Combine XY and angle features
        xy_files = list((self.output_path / 'xy_features/total').glob('*.csv'))
        angle_files = list((self.output_path / 'angle_features/total').glob('*.csv'))
        
        features_xy = pd.concat([pd.read_csv(f) for f in xy_files], ignore_index=True)
        features_angle = pd.concat([pd.read_csv(f) for f in angle_files], ignore_index=True)
        
        # Merge on video ID
        features_total = pd.merge(features_xy, features_angle, on='video', how='inner')
        
        # Apply feature selection (keep only limb features)
        keywords = ['wrist', 'ankle', 'elbow', 'knee']
        filtered_cols = [c for c in features_total.columns 
                        if any(k in c.lower() for k in keywords)]
        features_total = features_total[['video'] + filtered_cols]
        
        # Save consolidated features
        features_total.to_csv(self.output_path / 'features_total_consolidated.csv', index=False)
        
        self.log(f"Total features shape: {features_total.shape}")
        self.log(f"Features saved to {self.output_path / 'features_total_consolidated.csv'}")
        
        # Track final counts
        xy_ids = features_xy['video'].unique()
        angle_ids = features_angle['video'].unique()
        final_ids = features_total['video'].unique()
        
        self.track_data_loss("Feature Consolidation (XY)", xy_ids, final_ids)
        self.track_data_loss("Feature Consolidation (Angle)", angle_ids, final_ids)
    
    def run(self):
        """Execute full pipeline."""
        self.log("=" * 60)
        self.log("Starting Pose Processing Pipeline")
        self.log("=" * 60)
        
        # Setup
        self.setup_directories()
        vid_info = self.load_video_info()
        
        # Processing stages
        self.process_json_annotations(vid_info)
        self.smooth_and_normalize()
        self.compute_kinematics()
        self.extract_features()
        self.consolidate_features()
        
        # Generate reports
        self.save_data_loss_report()
        
        self.log("=" * 60)
        self.log("Pipeline completed successfully")
        self.log(f"Log file: {self.log_file}")
        self.log("=" * 60)


if __name__ == '__main__':
    # Configuration
    DATASET = 'PANDA2'
    BASE_PATH = './data2'
    N_WORKERS = 20
    OVERWRITE = True
    
    # Run pipeline
    pipeline = PoseProcessingPipeline(
        dataset=DATASET,
        base_path=BASE_PATH,
        n_workers=N_WORKERS,
        overwrite=OVERWRITE
    )
    
    pipeline.run()