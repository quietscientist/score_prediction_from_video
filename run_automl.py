#!/usr/bin/env python3
"""
AutoML Training and Inference for GMA Score Prediction
Integrates with pose processing pipeline for end-to-end prediction.
"""

import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    average_precision_score, confusion_matrix, 
    ConfusionMatrixDisplay, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance

import autosklearn
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import autosklearn.metrics

warnings.filterwarnings("ignore")


class GMAAutoMLPipeline:
    """AutoML pipeline for GMA score prediction with comprehensive logging."""
    
    def __init__(self, 
                 data_path: str = './data',
                 output_path: str = './automl_output',
                 feature_type: str = 'total',
                 feature_file: Optional[str] = None,
                 apply_exclusions: bool = True,
                 prereg: bool = False,
                 time_limit: int = 300,
                 per_run_limit: int = 30):
        """
        Initialize AutoML pipeline.
        
        Args:
            data_path: Path to data directory (for splits, scores, exclusions)
            output_path: Path for outputs
            feature_type: 'total' or 'windows' features
            feature_file: Path to feature file (if None, auto-detects from pose pipeline output)
            apply_exclusions: Whether to apply exclusions from all_excluded_videos.csv
            prereg: Whether to apply pre-registered model feature name mapping
            time_limit: Total time for AutoML (seconds)
            per_run_limit: Time per model run (seconds)
        """
        self.data_path = Path(data_path)
        self.feature_type = feature_type
        self.feature_file = Path(feature_file) if feature_file else None
        self.apply_exclusions = apply_exclusions
        self.prereg = prereg
        self.time_limit = time_limit
        self.per_run_limit = per_run_limit
        
        # If prereg is enabled, automatically save to 'prereg' subfolder
        if prereg:
            self.output_path = Path(output_path) / 'prereg'
        else:
            self.output_path = Path(output_path)
        
        # Setup directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_path / 'logs'
        self.log_path.mkdir(exist_ok=True)
        self.plots_path = self.output_path / 'plots'
        self.plots_path.mkdir(exist_ok=True)
        self.models_path = self.output_path / 'models'
        self.models_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self.log_file = self.log_path / f'automl_{datetime.now():%Y%m%d_%H%M%S}.log'
        self.data_loss_log = {
            'stage': [],
            'expected_count': [],
            'actual_count': [],
            'lost_count': [],
            'lost_ids': [],
            'timestamp': []
        }
        
        # Feature name mapping (new -> old format for pre-registered models)
        self.feature_mapping = {
            "Cross-corr_wrist_pos": "Wrist_lrCorr_x",
            "Cross-corr_knee_angle": "Knee_lrCorr_angle",
            "Med_elbow_angle_vel": "Elbow_median_vel_angle",
            "Med_wrist_pos_x": "Wrist_medianx",
            "Med_wrist_pos_y": "Wrist_mediany",
            "Cross-corr_elbow_angle": "Elbow_lrCorr_angle",
            "Med_wrist_angle_vel": "Wrist_median_vel_angle",
            "Med_ankle_pos_x": "Ankle_medianx",
            "Med_ankle_pos_y": "Ankle_mediany",
            "Med_knee_angle_vel": "Knee_median_vel_angle",
            "Entropy_elbow_angle": "Elbow_entropy_angle",
            "Med_wrist_vel_x": "Wrist_medianvelx",
            "IQR_wrist_vel_x": "Wrist_IQRvelx",
            "IQR_ankle_vel_y": "Ankle_IQRvely",
            "IQR_wrist_pos_x": "Wrist_IQRx",
            "IQR_wrist_pos_y": "Wrist_IQRy",
            "IQR_wrist_accel_x": "Wrist_IQRaccx",
            "Mean_elbow_angle": "Elbow_mean_angle",
            "IQR_elbow_angle_vel": "Elbow_IQR_vel_angle",
            "IQR_elbow_angle_accel": "Elbow_IQR_acc_angle",
            "IQR_ankle_vel_x": "Ankle_IQRvelx",
            "Med_ankle_vel_y": "Ankle_medianvely",
            "Cross-corr_ankle_pos": "Ankle_lrCorr_x",
            "Med_ankle_vel_x": "Ankle_medianvelx",
            "IQR_knee_angle_vel": "Knee_IQR_vel_angle",
            "IQR_ankle_pos_y": "Ankle_IQRy",
            "Mean_knee_angle": "Knee_mean_angle",
            "IQR_knee_angle_accel": "Knee_IQR_acc_angle",
            "Entropy_ankle_pos": "Ankle_meanent",
            "Entropy_wrist_pos": "Wrist_meanent",
            "Stdev_elbow_angle": "Elbow_stdev_angle",
            "IQR_ankle_pos_x": "Ankle_IQRx",
            "Entropy_knee_angle": "Knee_entropy_angle",
            "IQR_wrist_accel_y": "Wrist_IQRaccy",
            "IQR_wrist_vel_y": "Wrist_IQRvely",
            "Stdev_knee_angle": "Knee_stdev_angle",
            "Med_wrist_vel_y": "Wrist_medianvely",
            "IQR_ankle_accel_x": "Ankle_IQRaccx",
            "IQR_ankle_accel_y": "Ankle_IQRaccy"
        }
        
        self.automl = None
        self.log("AutoML Pipeline initialized")
        self.log(f"Exclusions: {'ENABLED' if apply_exclusions else 'DISABLED (--no-exclusions flag)'}")
        self.log(f"Pre-registered model mapping: {'ENABLED' if prereg else 'DISABLED'}")
        if prereg:
            self.log(f"Output directory (prereg mode): {self.output_path}")
    
    def log(self, message: str):
        """Write to log file with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_msg)
        print(message)
    
    def track_data_loss(self, stage: str, expected_ids, actual_ids, 
                       context: Optional[str] = None):
        """Track data loss at each stage."""
        expected_set = set(str(x) for x in expected_ids)
        actual_set = set(str(x) for x in actual_ids)
        lost = expected_set - actual_set
        
        self.data_loss_log['stage'].append(stage)
        self.data_loss_log['expected_count'].append(len(expected_set))
        self.data_loss_log['actual_count'].append(len(actual_set))
        self.data_loss_log['lost_count'].append(len(lost))
        self.data_loss_log['lost_ids'].append(list(lost))
        self.data_loss_log['timestamp'].append(datetime.now())
        
        loss_pct = len(lost) / len(expected_set) * 100 if expected_set else 0
        
        msg = f"{stage}: Expected {len(expected_set)} | Got {len(actual_set)} | Lost {len(lost)} ({loss_pct:.1f}%)"
        if context:
            msg += f" - {context}"
        self.log(msg)
        
        if lost and len(lost) <= 10:
            self.log(f"  Lost IDs: {sorted(list(lost))}")
        elif lost:
            self.log(f"  Lost IDs (first 10): {sorted(list(lost))[:10]}")
    
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot absolute counts
        x = range(len(df))
        ax1.plot(x, df['expected_count'], 'o-', label='Expected', 
                linewidth=2, markersize=8, color='#2ecc71')
        ax1.plot(x, df['actual_count'], 'o-', label='Actual', 
                linewidth=2, markersize=8, color='#3498db')
        ax1.fill_between(x, df['actual_count'], df['expected_count'], 
                         alpha=0.3, color='#e74c3c')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['stage'], rotation=45, ha='right')
        ax1.set_ylabel('Sample Count', fontsize=12)
        ax1.set_title('Data Flow Through AutoML Pipeline', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot loss percentage
        loss_pct = (df['lost_count'] / df['expected_count'] * 100).fillna(0)
        bars = ax2.bar(x, loss_pct, color='#e74c3c', alpha=0.6, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars, loss_pct)):
            if pct > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['stage'], rotation=45, ha='right')
        ax2.set_ylabel('Loss (%)', fontsize=12)
        ax2.set_title('Data Loss Percentage by Stage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'data_loss_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log("Data loss visualization saved")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                  pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all required data with loss tracking."""
        self.log("="*70)
        self.log("Loading data...")
        self.log("="*70)
        
        # Load split definitions
        train_ids = pd.read_csv(self.data_path / 'train.csv')
        val_ids = pd.read_csv(self.data_path / 'val.csv')
        test_ids = pd.read_csv(self.data_path / 'test.csv')
        test_holdout_ids = pd.read_csv(self.data_path / 'holdout.csv')  # Renamed from holdout
        
        self.log(f"Original split sizes:")
        self.log(f"  Train: {len(train_ids)}, Val: {len(val_ids)} → Combined Train: {len(train_ids) + len(val_ids)}")
        self.log(f"  Test: {len(test_ids)}")
        self.log(f"  Test_Holdout (LockBox): {len(test_holdout_ids)}")
        
        # Load exclusions (TAKES PRECEDENCE if enabled)
        excluded_ids = pd.DataFrame()
        exclusion_file = self.data_path / 'all_excluded_videos.csv'
        
        if not self.apply_exclusions:
            self.log(f"\n⚠️ EXCLUSIONS DISABLED (--no-exclusions flag)")
            self.log(f"  Skipping {exclusion_file}")
            self.log(f"  All videos will be included (except those without features/scores)")
        elif exclusion_file.exists():
            excluded_ids = pd.read_csv(exclusion_file)
            self.log(f"\n⚠️ EXCLUSIONS LOADED: {len(excluded_ids)} videos will be excluded")
            
            # Check what columns exist
            self.log(f"  Exclusion file columns: {list(excluded_ids.columns)}")
            
            # Determine the ID column name
            id_col = None
            for possible_name in ['gma_id', 'infant', 'video', 'id', 'video_id']:
                if possible_name in excluded_ids.columns:
                    id_col = possible_name
                    break
            
            if id_col is None:
                self.log("  WARNING: Could not identify ID column in exclusion file")
                self.log(f"  Available columns: {list(excluded_ids.columns)}")
            else:
                excluded_set = set(excluded_ids[id_col])
                
                # Check overlap with splits
                all_split_ids = (set(train_ids['gma_id']) | set(val_ids['gma_id']) | 
                               set(test_ids['gma_id']) | set(test_holdout_ids['gma_id']))
                overlap = excluded_set & all_split_ids
                
                if overlap:
                    self.log(f"  ⚠️ {len(overlap)} excluded IDs are in train/val/test/test_holdout")
                    self.log(f"  These will be removed (exclusion takes precedence)")
                    self.log(f"  Excluded from splits: {sorted(list(overlap))[:10]}...")
                else:
                    self.log(f"  ✓ No excluded IDs found in splits")
        else:
            self.log(f"\nNote: No exclusion file found at {exclusion_file}")
        
        # Check for overlaps between splits
        self._check_split_overlaps(train_ids, val_ids, test_ids, test_holdout_ids)
        
        # Load scores
        scores = pd.read_csv(self.data_path / 'gma_score_prediction_scores.csv')
        
        # Normalize scores to 0-indexed
        if scores['score'].min() != 0:
            original_scores = scores['score'].copy()
            scores = scores[scores['score'].isin([1, 2, 3])].copy()
            scores['score'] = scores['score'].apply(lambda x: int(x) - 1)
            self.log(f"\nNormalized scores from {original_scores.unique()} to {scores['score'].unique()}")
        
        # Load features
        features = self._load_features()
        
        self.log(f"\nLoaded {self.feature_type} features: {features.shape}")
        self.log(f"Feature columns: {list(features.columns[:5])}... ({len(features.columns)} total)")
        
        return train_ids, val_ids, test_ids, test_holdout_ids, excluded_ids, scores, features
    
    def _load_features(self) -> pd.DataFrame:
        """Load features from specified file or auto-detect from pose pipeline output."""
        
        # If explicit feature file provided, use it
        if self.feature_file and self.feature_file.exists():
            self.log(f"\nLoading features from: {self.feature_file}")
            features = pd.read_csv(self.feature_file)
        else:
            # Auto-detect from pose pipeline output
            self.log("\nAuto-detecting feature file from pose pipeline output...")
            
            # Try standard pose pipeline output location
            pose_output_paths = [
                Path(f'./pose_estimates/PANDA2_pose_estimates/features_total_consolidated.csv'),
                Path(f'./pose_estimates/*/features_total_consolidated.csv'),
                self.data_path / 'final_total_features.csv',  # Fallback to old location
                self.data_path / 'final_window_features.csv',
            ]
            
            feature_file_found = None
            for path_pattern in pose_output_paths:
                # Handle glob patterns
                if '*' in str(path_pattern):
                    matches = list(Path('.').glob(str(path_pattern)))
                    if matches:
                        feature_file_found = matches[0]
                        break
                elif path_pattern.exists():
                    feature_file_found = path_pattern
                    break
            
            if feature_file_found:
                self.log(f"  Found features at: {feature_file_found}")
                features = pd.read_csv(feature_file_found)
            else:
                # Last resort: try legacy naming
                self.log("  Could not auto-detect pose pipeline output, trying legacy locations...")
                if self.feature_type == 'windows':
                    legacy_path = self.data_path / 'final_window_features.csv'
                else:
                    legacy_path = self.data_path / 'final_total_features.csv'
                
                if legacy_path.exists():
                    self.log(f"  Found features at: {legacy_path}")
                    features = pd.read_csv(legacy_path)
                else:
                    raise FileNotFoundError(
                        f"Could not find feature file. Tried:\n" +
                        "\n".join([f"  - {p}" for p in pose_output_paths]) +
                        f"\n  - {legacy_path}\n\n" +
                        "Please specify feature file explicitly with --feature-file argument"
                    )
        
        # Clean column names
        if 'Unnamed: 0' in features.columns:
            features = features.drop(columns=['Unnamed: 0'])
        
        # Rename 'video' column to 'infant' if needed for consistency
        if 'video' in features.columns and 'infant' not in features.columns:
            features = features.rename(columns={'video': 'infant'})
            self.log("  Renamed 'video' column to 'infant' for consistency")
        
        return features
    
    def _check_split_overlaps(self, train_ids, val_ids, test_ids, test_holdout_ids):
        """Check for overlaps between data splits."""
        self.log("\nChecking for overlaps between splits:")
        
        splits = {
            'Train': set(train_ids['gma_id']),
            'Val': set(val_ids['gma_id']),
            'Test': set(test_ids['gma_id']),
            'Test_Holdout': set(test_holdout_ids['gma_id'])
        }
        
        overlaps_found = False
        for name1, ids1 in splits.items():
            for name2, ids2 in splits.items():
                if name1 < name2:  # Only check each pair once
                    overlap = ids1 & ids2
                    if overlap:
                        self.log(f"  WARNING: {name1}-{name2} overlap: {len(overlap)} IDs")
                        overlaps_found = True
                    else:
                        self.log(f"  {name1}-{name2}: No overlap ✓")
        
        if not overlaps_found:
            self.log("  All splits are properly separated ✓")
    
    def _average_left_right_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Average left and right side features for pre-registered model compatibility."""
        self.log("\n  Averaging left/right side features...")
        
        # Start with ID column
        id_col = 'infant' if 'infant' in features.columns else 'video'
        averaged_features = {id_col: features[id_col]}
        
        # Find all unique feature patterns
        all_cols = [col for col in features.columns if col not in ['infant', 'video']]
        
        # Track which columns we've processed
        processed = set()
        averaged_count = 0
        
        for col in all_cols:
            if col in processed:
                continue
            
            # Parse column name: metric_LSide or metric_RSide format
            if '_L' in col:
                # This is a left-side feature
                parts = col.split('_L')
                if len(parts) == 2:
                    metric = parts[0]
                    body_part = parts[1]
                    
                    left_col = f"{metric}_L{body_part}"
                    right_col = f"{metric}_R{body_part}"
                    
                    if left_col in features.columns and right_col in features.columns:
                        # Average left and right
                        avg_col_name = f"{metric}_{body_part}"
                        averaged_features[avg_col_name] = (features[left_col] + features[right_col]) / 2
                        processed.add(left_col)
                        processed.add(right_col)
                        averaged_count += 1
                    else:
                        # Only left exists
                        averaged_features[col] = features[col]
                        processed.add(col)
            
            elif '_R' in col:
                # This is a right-side feature
                parts = col.split('_R')
                if len(parts) == 2:
                    metric = parts[0]
                    body_part = parts[1]
                    
                    left_col = f"{metric}_L{body_part}"
                    right_col = f"{metric}_R{body_part}"
                    
                    if left_col not in features.columns:
                        # Only right exists (left was already processed or doesn't exist)
                        if right_col not in processed:
                            averaged_features[col] = features[col]
                            processed.add(col)
            
            else:
                # Not a left/right feature (e.g., lrCorr_x_Ankle)
                averaged_features[col] = features[col]
                processed.add(col)
        
        result_df = pd.DataFrame(averaged_features)
        
        self.log(f"    Averaged {averaged_count} left/right feature pairs")
        self.log(f"    Total features after averaging: {len(result_df.columns) - 1} (excluding ID column)")
        
        return result_df
    
    def _apply_prereg_name_mapping(self, features: pd.DataFrame) -> pd.DataFrame:
        """Map averaged feature names to pre-registered model format."""
        self.log("\n  Applying pre-registered model name mapping...")
        
        # Mapping from averaged format to pre-registered format
        # Format: averaged_name → prereg_name
        prereg_mapping = {
            # Ankle position features (after averaging L/R)
            'IQRx_Ankle': 'Ankle_IQRx',
            'IQRy_Ankle': 'Ankle_IQRy',
            'medianx_Ankle': 'Ankle_medianx',
            'mediany_Ankle': 'Ankle_mediany',
            'meanent_Ankle': 'Ankle_meanent',
            'medianvelx_Ankle': 'Ankle_medianvelx',
            'medianvely_Ankle': 'Ankle_medianvely',
            'IQRvelx_Ankle': 'Ankle_IQRvelx',
            'IQRvely_Ankle': 'Ankle_IQRvely',
            'IQRaccx_Ankle': 'Ankle_IQRaccx',
            'IQRaccy_Ankle': 'Ankle_IQRaccy',
            
            # Wrist position features (after averaging L/R)
            'IQRx_Wrist': 'Wrist_IQRx',
            'IQRy_Wrist': 'Wrist_IQRy',
            'medianx_Wrist': 'Wrist_medianx',
            'mediany_Wrist': 'Wrist_mediany',
            'meanent_Wrist': 'Wrist_meanent',
            'medianvelx_Wrist': 'Wrist_medianvelx',
            'medianvely_Wrist': 'Wrist_medianvely',
            'IQRvelx_Wrist': 'Wrist_IQRvelx',
            'IQRvely_Wrist': 'Wrist_IQRvely',
            'IQRaccx_Wrist': 'Wrist_IQRaccx',
            'IQRaccy_Wrist': 'Wrist_IQRaccy',
            
            # Elbow angle features (after averaging L/R)
            'IQR_acc_angle_Elbow': 'Elbow_IQR_acc_angle',
            'IQR_vel_angle_Elbow': 'Elbow_IQR_vel_angle',
            'entropy_angle_Elbow': 'Elbow_entropy_angle',
            'mean_angle_Elbow': 'Elbow_mean_angle',
            'median_vel_angle_Elbow': 'Elbow_median_vel_angle',
            'stdev_angle_Elbow': 'Elbow_stdev_angle',
            
            # Knee angle features (after averaging L/R)
            'IQR_acc_angle_Knee': 'Knee_IQR_acc_angle',
            'IQR_vel_angle_Knee': 'Knee_IQR_vel_angle',
            'entropy_angle_Knee': 'Knee_entropy_angle',
            'mean_angle_Knee': 'Knee_mean_angle',
            'median_vel_angle_Knee': 'Knee_median_vel_angle',
            'stdev_angle_Knee': 'Knee_stdev_angle',
            
            # Cross-correlation features (these don't have L/R, already averaged internally)
            'lrCorr_x_Ankle': 'Ankle_lrCorr_x',
            'lrCorr_x_Wrist': 'Wrist_lrCorr_x',
            # 'lrCorr_x_Elbow': 'Elbow_lrCorr_x',  # Added in case it exists
            # 'lrCorr_x_Knee': 'Knee_lrCorr_x',    # Added in case it exists
            'lrCorr_angle_Elbow': 'Elbow_lrCorr_angle',
            'lrCorr_angle_Knee': 'Knee_lrCorr_angle',
        }
        
        # Apply mapping
        mapped_count = 0
        for old_name, new_name in prereg_mapping.items():
            if old_name in features.columns:
                mapped_count += 1
        
        features = features.rename(columns=prereg_mapping)
        
        self.log(f"    Mapped {mapped_count} feature names to pre-registered format")
        
        # Check for any unmapped features (excluding ID columns)
        unmapped = [col for col in features.columns 
                   if col not in ['infant', 'video'] 
                   and col not in prereg_mapping.values()]
        
        if unmapped:
            self.log(f"    ⚠️ Warning: {len(unmapped)} features were not mapped:")
            self.log(f"    Unmapped features: {sorted(unmapped)[:10]}")
            if len(unmapped) > 10:
                self.log(f"      ... and {len(unmapped) - 10} more")
        
        # Final feature count verification
        feature_cols = [col for col in features.columns if col not in ['infant', 'video']]
        self.log(f"\n  ✓ Final feature count: {len(feature_cols)} features")
        
        if len(feature_cols) != 38:
            self.log(f"    ⚠️ WARNING: Expected 38 features but got {len(feature_cols)}!")
            self.log(f"    Features present: {sorted(feature_cols)}")
        else:
            self.log(f"    ✓ Confirmed: Exactly 38 features as expected for pre-registered model")
        
        return features
    
    def prepare_datasets(self, train_ids, val_ids, test_ids, test_holdout_ids, 
                        excluded_ids, scores, features) -> Dict:
        """Prepare train/test/lockbox datasets with comprehensive loss tracking."""
        self.log("\n" + "="*70)
        self.log("Preparing datasets...")
        self.log("="*70)
        
        # Apply pre-registered feature mapping if enabled
        if self.prereg:
            self.log("\n🔄 Pre-registered model preprocessing...")
            self.log(f"  Features BEFORE preprocessing: {features.shape}")
            self.log(f"  Sample columns: {list(features.columns[:10])}")
            
            # Step 1: Average left and right side features
            features = self._average_left_right_features(features)
            
            self.log(f"  Features AFTER averaging L/R: {features.shape}")
            self.log(f"  Sample columns: {list(features.columns[:10])}")
            
            # Step 2: Apply feature name mapping to match pre-registered model format
            features = self._apply_prereg_name_mapping(features)
            
            self.log(f"  Features AFTER name mapping: {features.shape}")
            self.log(f"  Final columns: {list(features.columns[:10])}")
        
        # Track where each LockBox ID goes (detailed forensics)
        lockbox_tracking = self._initialize_lockbox_tracking(test_holdout_ids, features, scores, excluded_ids)
        
        # Check for NaNs in features
        self._log_nan_stats(features, "Raw features")
        
        # CRITICAL: Apply exclusions FIRST (takes absolute precedence)
        # Can be disabled with --no-exclusions flag
        if not excluded_ids.empty and self.apply_exclusions:
            # Determine ID column in exclusion file
            id_col = None
            for possible_name in ['gma_id', 'infant', 'video', 'id', 'video_id']:
                if possible_name in excluded_ids.columns:
                    id_col = possible_name
                    break
            
            if id_col:
                excluded_set = set(excluded_ids[id_col])
                before_exclusion = len(features)
                
                # Track which lockbox IDs were excluded
                lockbox_excluded = set(test_holdout_ids['gma_id']) & excluded_set
                for vid in lockbox_excluded:
                    lockbox_tracking[vid]['excluded'] = True
                    if 'reason' in excluded_ids.columns:
                        reason_row = excluded_ids[excluded_ids[id_col] == vid]
                        if not reason_row.empty:
                            lockbox_tracking[vid]['exclusion_reason'] = reason_row['reason'].iloc[0]
                
                # Remove excluded videos from features
                features = features[~features['infant'].isin(excluded_set)].copy()
                excluded_count = before_exclusion - len(features)
                
                self.log(f"\n⚠️ EXCLUSION APPLIED: Removed {excluded_count} videos from features")
                if excluded_count > 0:
                    actually_excluded = excluded_set - set(features['infant'])
                    self.log(f"  Videos excluded from features: {len(actually_excluded)}")
                    self.log(f"  Excluded IDs (first 10): {sorted(list(actually_excluded))[:10]}")
                    
                    # Log lockbox-specific exclusions
                    if lockbox_excluded:
                        self.log(f"  ⚠️ LockBox IDs excluded: {len(lockbox_excluded)}")
                        self.log(f"     IDs: {sorted(list(lockbox_excluded))[:10]}")
                
                # Also remove from scores
                before_score_exclusion = len(scores)
                scores = scores[~scores['infant'].isin(excluded_set)].copy()
                score_excluded_count = before_score_exclusion - len(scores)
                if score_excluded_count > 0:
                    self.log(f"  Videos excluded from scores: {score_excluded_count}")
        elif not self.apply_exclusions:
            self.log(f"\n✓ EXCLUSIONS SKIPPED (--no-exclusions flag enabled)")
            self.log(f"  All {len(features)} videos with features will be processed")
        
        # Track which lockbox IDs made it past exclusions
        lockbox_after_exclusion = set(test_holdout_ids['gma_id']) & set(features['infant'])
        for vid in lockbox_after_exclusion:
            lockbox_tracking[vid]['in_features_after_exclusion'] = True
        
        # Merge features with scores (only after exclusions applied)
        original_feature_ids = features['infant'].unique()
        features_with_scores = pd.merge(features, scores, on='infant', how='inner')
        
        # Track which lockbox IDs have scores
        lockbox_with_scores = set(test_holdout_ids['gma_id']) & set(features_with_scores['infant'])
        lockbox_no_scores = lockbox_after_exclusion - lockbox_with_scores
        for vid in lockbox_no_scores:
            lockbox_tracking[vid]['missing_score'] = True
        for vid in lockbox_with_scores:
            lockbox_tracking[vid]['has_score'] = True
        
        self.track_data_loss(
            "Feature-Score Merge (after exclusions)",
            original_feature_ids,
            features_with_scores['infant'].unique(),
            "Missing scores after exclusion"
        )
        
        # Handle missing values
        before_dropna = len(features_with_scores)

        # Define columns that are actually needed for modeling
        feature_cols = [col for col in features_with_scores.columns 
                        if col not in ['infant', 'age_corrected', 'age_chronological']]
        required_cols = feature_cols + ['score']  # Only check features + score

        # Drop rows with NaN only in required columns
        features_clean = features_with_scores.dropna(subset=required_cols)
        
        # Track which lockbox IDs were dropped due to NaN
        lockbox_before_dropna = set(test_holdout_ids['gma_id']) & set(features_with_scores['infant'])
        lockbox_after_dropna = set(test_holdout_ids['gma_id']) & set(features_clean['infant'])
        lockbox_dropped_nan = lockbox_before_dropna - lockbox_after_dropna
        for vid in lockbox_dropped_nan:
            lockbox_tracking[vid]['dropped_nan'] = True
            # Find which columns had NaN for this video
            vid_row = features_with_scores[features_with_scores['infant'] == vid]
            if not vid_row.empty:
                nan_cols = vid_row.columns[vid_row.isnull().any()].tolist()
                lockbox_tracking[vid]['nan_columns'] = nan_cols
        
        if before_dropna > len(features_clean):
            dropped_count = before_dropna - len(features_clean)
            self.log(f"\nDropped {dropped_count} rows with missing values ({dropped_count/before_dropna*100:.1f}%)")
            
            if lockbox_dropped_nan:
                self.log(f"  ⚠️ LockBox IDs dropped due to NaN: {len(lockbox_dropped_nan)}")
                self.log(f"     IDs: {sorted(list(lockbox_dropped_nan))}")
            
            missing_counts = features_with_scores.isnull().sum()
            top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
            self.log("Top 5 columns with missing values:")
            for col, count in top_missing.items():
                self.log(f"  {col}: {count} missing ({count/before_dropna*100:.1f}%)")
        
        # Apply preprocessing based on feature type
        if self.feature_type == 'total':
            self.log("\nApplying preprocessing for 'total' features:")
            self.log(f"  Initial samples: {len(features_clean)}")
            len_orig = len(features_clean)
            # Merge abnormal scores (2 -> 1)
            # features_clean['score'] = features_clean['score'].replace({2: 1})
            # self.log(f"  Merged score 2 into score 1 (Abnormal category)")

            #drop atypical scores (2)
            features_clean = features_clean[features_clean['score'] != 2]
            self.log(f"  Removed {len_orig - len(features_clean)} score 2 (Atypical category)")
            
            score_dist = features_clean['score'].value_counts().sort_index()
            self.log(f"  Final score distribution: {dict(score_dist)}")
        
        # Split into Train (train+val), Test, LockBox (test_holdout)
        datasets = {}
        
        # Training set (train + val COMBINED - AutoML doesn't need separate validation)
        train_data = features_clean[
            features_clean['infant'].isin(train_ids['gma_id']) | 
            features_clean['infant'].isin(val_ids['gma_id'])
        ]
        self.track_data_loss(
            "Train Split (train+val combined)",
            list(train_ids['gma_id']) + list(val_ids['gma_id']),
            train_data['infant'].unique(),
            "Missing features or scores"
        )
        
        # Test set
        test_data = features_clean[features_clean['infant'].isin(test_ids['gma_id'])]
        self.track_data_loss(
            "Test Split",
            test_ids['gma_id'],
            test_data['infant'].unique(),
            "Missing features or scores"
        )
        
        # LockBox set (formerly holdout, now test_holdout)
        lockbox_data = features_clean[features_clean['infant'].isin(test_holdout_ids['gma_id'])]
        
        # Mark which lockbox IDs made it to final dataset
        final_lockbox_ids = set(lockbox_data['infant'])
        for vid in final_lockbox_ids:
            lockbox_tracking[vid]['in_final_dataset'] = True
        
        self.track_data_loss(
            "LockBox Split (test_holdout)",
            test_holdout_ids['gma_id'],
            lockbox_data['infant'].unique(),
            "Missing features or scores"
        )
        
        # Save filtered IDs and scores
        self._save_filtered_splits(train_data, test_data, lockbox_data)
        
        # ADD IT HERE - INSIDE prepare_datasets, AFTER train_data/test_data/lockbox_data are created
        self._save_comprehensive_split_summary(
            train_data, test_data, lockbox_data,
            train_ids, val_ids, test_ids, test_holdout_ids,
            excluded_ids, scores
        )

        # Generate detailed LockBox forensics report
        self._log_lockbox_forensics(lockbox_tracking, test_holdout_ids)
        
        # Prepare X, y arrays
        drop_cols = ['infant', 'score']
        if 'age_corrected' in features_clean.columns:
            drop_cols.append('age_corrected')
        if 'age_chronological' in features_clean.columns:
            drop_cols.append('age_chronological')
        
        datasets['X_train'] = train_data.drop(columns=drop_cols)
        datasets['y_train'] = train_data['score']
        datasets['train_ids'] = train_data['infant']
        
        datasets['X_test'] = test_data.drop(columns=drop_cols)
        datasets['y_test'] = test_data['score']
        datasets['test_ids'] = test_data['infant']
        
        datasets['X_lockbox'] = lockbox_data.drop(columns=drop_cols)
        datasets['y_lockbox'] = lockbox_data['score']
        datasets['lockbox_ids'] = lockbox_data['infant']
        
        features_to_drop = ['lrCorr_x_Elbow', 'lrCorr_x_Knee']
        # Only drop if they exist
        features_to_drop = [f for f in features_to_drop if f in datasets['X_train'].columns]
        if features_to_drop:
            self.log(f"\n🗑️ Dropping {len(features_to_drop)} lrCorr_x features: {features_to_drop}")
            datasets['X_train'] = datasets['X_train'].drop(columns=features_to_drop)
            datasets['X_test'] = datasets['X_test'].drop(columns=features_to_drop)
            datasets['X_lockbox'] = datasets['X_lockbox'].drop(columns=features_to_drop)
            self.log(f"  Remaining features: {datasets['X_train'].shape[1]}")

        # Log final sizes and NaN stats
        self.log("\nFinal dataset sizes:")
        self.log(f"  Train (train+val): {len(datasets['X_train'])} samples, {datasets['X_train'].shape[1]} features")
        self.log(f"  Test: {len(datasets['X_test'])} samples")
        self.log(f"  LockBox: {len(datasets['X_lockbox'])} samples")
        
        # Check NaNs in final datasets
        self._log_nan_stats(datasets['X_train'], "Train features")
        self._log_nan_stats(datasets['X_test'], "Test features")
        self._log_nan_stats(datasets['X_lockbox'], "LockBox features")
        
        # Log class distributions
        for split in ['train', 'test', 'lockbox']:
            y = datasets[f'y_{split}']
            dist = y.value_counts().sort_index()
            self.log(f"  {split.capitalize()} score distribution: {dict(dist)}")
        
        # Save filtered IDs and scores
        self._save_filtered_splits(train_data, test_data, lockbox_data)

        

        
        # Save lockbox tracking report
        self._save_lockbox_tracking_report(lockbox_tracking)
        
        return datasets
    
    def _initialize_lockbox_tracking(self, test_holdout_ids, features, scores, excluded_ids) -> Dict:
        """Initialize tracking dictionary for each LockBox ID."""
        tracking = {}
        
        feature_ids = set(features['infant']) if 'infant' in features.columns else set()
        score_ids = set(scores['infant']) if 'infant' in scores.columns else set()
        
        # Determine exclusion ID column
        excluded_set = set()
        if not excluded_ids.empty:
            for possible_name in ['gma_id', 'infant', 'video', 'id', 'video_id']:
                if possible_name in excluded_ids.columns:
                    excluded_set = set(excluded_ids[possible_name])
                    break
        
        for vid in test_holdout_ids['gma_id']:
            tracking[vid] = {
                'in_original_features': vid in feature_ids,
                'in_original_scores': vid in score_ids,
                'excluded': vid in excluded_set,
                'exclusion_reason': None,
                'in_features_after_exclusion': False,
                'has_score': False,
                'missing_score': False,
                'dropped_nan': False,
                'nan_columns': [],
                'in_final_dataset': False,
            }
        
        return tracking
    
    def _log_lockbox_forensics(self, lockbox_tracking: Dict, test_holdout_ids):
        """Generate detailed forensics report for LockBox IDs."""
        self.log("\n" + "="*70)
        self.log("LOCKBOX ID FORENSICS")
        self.log("="*70)
        
        total_lockbox = len(test_holdout_ids)
        final_count = sum(1 for v in lockbox_tracking.values() if v['in_final_dataset'])
        lost_count = total_lockbox - final_count
        
        self.log(f"Total LockBox IDs: {total_lockbox}")
        self.log(f"Made it to final dataset: {final_count}")
        self.log(f"Lost during pipeline: {lost_count} ({lost_count/total_lockbox*100:.1f}%)")
        
        # Categorize losses
        never_in_features = [vid for vid, info in lockbox_tracking.items() 
                            if not info['in_original_features']]
        excluded_ids = [vid for vid, info in lockbox_tracking.items() 
                       if info['excluded']]
        missing_scores = [vid for vid, info in lockbox_tracking.items() 
                         if not info['excluded'] and info['in_features_after_exclusion'] 
                         and info['missing_score']]
        dropped_nan = [vid for vid, info in lockbox_tracking.items() 
                      if info['dropped_nan']]
        
        self.log("\nLoss breakdown:")
        self.log(f"  1. Never in features file: {len(never_in_features)} IDs")
        if never_in_features:
            self.log(f"     (Pose processing failed or features never generated)")
            self.log(f"     IDs: {sorted(never_in_features)[:10]}{'...' if len(never_in_features) > 10 else ''}")
        
        self.log(f"  2. Excluded (all_excluded_videos.csv): {len(excluded_ids)} IDs")
        if excluded_ids:
            # Group by exclusion reason
            reasons = {}
            for vid in excluded_ids:
                reason = lockbox_tracking[vid]['exclusion_reason'] or 'unknown'
                reasons.setdefault(reason, []).append(vid)
            
            for reason, ids in reasons.items():
                self.log(f"     Reason '{reason}': {len(ids)} IDs")
                self.log(f"       IDs: {sorted(ids)[:10]}{'...' if len(ids) > 10 else ''}")
        
        self.log(f"  3. Missing GMA scores: {len(missing_scores)} IDs")
        if missing_scores:
            self.log(f"     (Had features but no score in gma_score_prediction_scores.csv)")
            self.log(f"     IDs: {sorted(missing_scores)[:10]}{'...' if len(missing_scores) > 10 else ''}")
        
        self.log(f"  4. Dropped due to NaN values: {len(dropped_nan)} IDs")
        if dropped_nan:
            self.log(f"     (Had features and scores but contained missing values)")
            # Show which features had NaN for first few IDs
            for vid in sorted(dropped_nan)[:5]:
                nan_cols = lockbox_tracking[vid]['nan_columns']
                self.log(f"     ID {vid}: NaN in {len(nan_cols)} columns: {nan_cols[:3]}{'...' if len(nan_cols) > 3 else ''}")
        
        # Summary table
        self.log("\nPer-ID status summary (first 20 LockBox IDs):")
        self.log(f"{'ID':<8} {'InFeatures':<12} {'Excluded':<10} {'HasScore':<10} {'DroppedNaN':<12} {'Final':<8}")
        self.log("-" * 70)
        
        for vid in sorted(lockbox_tracking.keys())[:20]:
            info = lockbox_tracking[vid]
            self.log(
                f"{vid:<8} "
                f"{'✓' if info['in_original_features'] else '✗':<12} "
                f"{'✓' if info['excluded'] else '✗':<10} "
                f"{'✓' if info['has_score'] else '✗':<10} "
                f"{'✓' if info['dropped_nan'] else '✗':<12} "
                f"{'✓' if info['in_final_dataset'] else '✗':<8}"
            )
        
        if len(lockbox_tracking) > 20:
            self.log(f"... (showing 20 of {len(lockbox_tracking)} total LockBox IDs)")
        
        self.log("="*70 + "\n")
    
    def _save_lockbox_tracking_report(self, lockbox_tracking: Dict):
        """Save detailed LockBox tracking report to CSV."""
        tracking_data = []
        
        for vid, info in lockbox_tracking.items():
            tracking_data.append({
                'lockbox_id': vid,
                'in_original_features': info['in_original_features'],
                'in_original_scores': info['in_original_scores'],
                'excluded': info['excluded'],
                'exclusion_reason': info['exclusion_reason'] or '',
                'in_features_after_exclusion': info['in_features_after_exclusion'],
                'has_score': info['has_score'],
                'missing_score': info['missing_score'],
                'dropped_nan': info['dropped_nan'],
                'nan_columns': ','.join(map(str, info['nan_columns'])),
                'in_final_dataset': info['in_final_dataset'],
            })
        
        df = pd.DataFrame(tracking_data)
        report_path = self.log_path / 'lockbox_id_tracking.csv'
        df.to_csv(report_path, index=False)
        
        self.log(f"LockBox ID tracking report saved to: {report_path}")
        
        # Also create summary stats
        summary = {
            'total_lockbox_ids': len(lockbox_tracking),
            'in_final_dataset': df['in_final_dataset'].sum(),
            'never_in_features': (~df['in_original_features']).sum(),
            'excluded': df['excluded'].sum(),
            'missing_score': df['missing_score'].sum(),
            'dropped_nan': df['dropped_nan'].sum(),
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = self.log_path / 'lockbox_tracking_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        self.log(f"LockBox tracking summary saved to: {summary_path}")
    
    def _save_filtered_splits(self, train_data, test_data, lockbox_data):
        """Save filtered IDs and scores as Train.csv, Test.csv, LockBox.csv"""
        self.log("\nSaving filtered splits (after exclusions)...")
        
        splits_output_path = self.output_path / 'filtered_splits'
        splits_output_path.mkdir(exist_ok=True)
        
        # Train (includes IDs and scores)
        train_export = train_data[['infant', 'score']].copy()
        train_export.columns = ['gma_id', 'score']
        train_export.to_csv(splits_output_path / 'Train.csv', index=False)
        self.log(f"  Saved Train.csv: {len(train_export)} samples")
        
        # Test
        test_export = test_data[['infant', 'score']].copy()
        test_export.columns = ['gma_id', 'score']
        test_export.to_csv(splits_output_path / 'Test.csv', index=False)
        self.log(f"  Saved Test.csv: {len(test_export)} samples")
        
        # LockBox
        lockbox_export = lockbox_data[['infant', 'score']].copy()
        lockbox_export.columns = ['gma_id', 'score']
        lockbox_export.to_csv(splits_output_path / 'LockBox.csv', index=False)
        self.log(f"  Saved LockBox.csv: {len(lockbox_export)} samples")
        
        self.log(f"  All filtered splits saved to: {splits_output_path}")

    def _save_comprehensive_split_summary(self, train_data, test_data, lockbox_data, 
                                      train_ids, val_ids, test_ids, test_holdout_ids, 
                                      excluded_ids, scores):
        """Save comprehensive summary of splits with original totals and exclusions."""
        self.log("\n" + "="*70)
        self.log("COMPREHENSIVE SPLIT SUMMARY")
        self.log("="*70)
        
        # Calculate totals
        total_original = len(train_ids) + len(val_ids) + len(test_ids) + len(test_holdout_ids)
        
        # Count exclusions
        n_excluded = 0
        if not excluded_ids.empty and self.apply_exclusions:
            # Determine ID column
            id_col = None
            for possible_name in ['gma_id', 'infant', 'video', 'id', 'video_id']:
                if possible_name in excluded_ids.columns:
                    id_col = possible_name
                    break
            if id_col:
                excluded_set = set(excluded_ids[id_col])
                all_split_ids = (set(train_ids['gma_id']) | set(val_ids['gma_id']) | 
                            set(test_ids['gma_id']) | set(test_holdout_ids['gma_id']))
                n_excluded = len(excluded_set & all_split_ids)
        
        # Get final counts and distributions
        def get_score_dist(data):
            """Get score distribution as percentages."""
            if len(data) == 0:
                return {'total': 0, 'pct_0': 0, 'pct_1': 0, 'pct_2': 0}
            
            dist = data['score'].value_counts()
            total = len(data)
            return {
                'total': total,
                'pct_0': (dist.get(0, 0) / total * 100) if total > 0 else 0,
                'pct_1': (dist.get(1, 0) / total * 100) if total > 0 else 0,
                'pct_2': (dist.get(2, 0) / total * 100) if total > 0 else 0,
            }
        
        train_dist = get_score_dist(train_data)
        test_dist = get_score_dist(test_data)
        holdout_dist = get_score_dist(lockbox_data)
        
        # Log summary
        self.log(f"\nTotal (original split definitions): {total_original}")
        self.log(f"Excluded: {n_excluded}")
        self.log(f"Train (train+val combined): {train_dist['total']} "
                f"(Score 0: {train_dist['pct_0']:.1f}%, "
                f"Score 1: {train_dist['pct_1']:.1f}%, "
                f"Score 2: {train_dist['pct_2']:.1f}%)")
        self.log(f"Test: {test_dist['total']} "
                f"(Score 0: {test_dist['pct_0']:.1f}%, "
                f"Score 1: {test_dist['pct_1']:.1f}%, "
                f"Score 2: {test_dist['pct_2']:.1f}%)")
        self.log(f"Holdout: {holdout_dist['total']} "
                f"(Score 0: {holdout_dist['pct_0']:.1f}%, "
                f"Score 1: {holdout_dist['pct_1']:.1f}%, "
                f"Score 2: {holdout_dist['pct_2']:.1f}%)")
        
        # Create summary dataframe
        summary_data = {
            'split': ['Total_Original', 'Excluded', 'Train', 'Test', 'Holdout'],
            'count': [
                total_original,
                n_excluded,
                train_dist['total'],
                test_dist['total'],
                holdout_dist['total']
            ],
            'score_0_pct': [
                None,
                None,
                train_dist['pct_0'],
                test_dist['pct_0'],
                holdout_dist['pct_0']
            ],
            'score_1_pct': [
                None,
                None,
                train_dist['pct_1'],
                test_dist['pct_1'],
                holdout_dist['pct_1']
            ],
            'score_2_pct': [
                None,
                None,
                train_dist['pct_2'],
                test_dist['pct_2'],
                holdout_dist['pct_2']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_path = self.output_path / 'comprehensive_split_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        self.log(f"\nComprehensive summary saved to: {summary_path}")
        self.log("="*70 + "\n")
        
        return summary_df

    def _log_nan_stats(self, df: pd.DataFrame, label: str):
        """Log NaN statistics for a dataframe."""
        total_values = df.shape[0] * df.shape[1]
        nan_count = df.isnull().sum().sum()
        nan_pct = (nan_count / total_values * 100) if total_values > 0 else 0
        
        self.log(f"{label}: {nan_pct:.2f}% NaN ({nan_count}/{total_values} values)")
        
        if nan_pct > 10:
            self.log(f"  ⚠️ WARNING: High NaN percentage!")
            top_nan = df.isnull().sum().sort_values(ascending=False).head(3)
            for col, count in top_nan.items():
                if count > 0:
                    col_pct = (count / len(df) * 100)
                    self.log(f"    {col}: {col_pct:.1f}% NaN")
    
    def train_model(self, X_train, y_train):
        """Train AutoML model."""
        self.log("\n" + "="*70)
        self.log("Training AutoML model...")
        self.log("="*70)
        self.log(f"Time limit: {self.time_limit}s total, {self.per_run_limit}s per run")
        
        # Fix pandas compatibility
        pd.DataFrame.iteritems = pd.DataFrame.items
        
        self.automl = AutoSklearn2Classifier(
            ensemble_size=1,
            dataset_compression=False,
            allow_string_features=False,
            time_left_for_this_task=self.time_limit,
            per_run_time_limit=self.per_run_limit,
            metric=autosklearn.metrics.balanced_accuracy,
            delete_tmp_folder_after_terminate=False,
            memory_limit=None,
            disable_evaluator_output=False,
        )
        
        self.log("Fitting AutoML model...")
        self.automl.fit(X_train, y_train)
        
        self.log("Refitting on full training data...")
        self.automl.refit(X_train.copy(), y_train.copy())
        
        # Save model
        model_path = self.models_path / f'automl_{self.feature_type}_{datetime.now():%Y%m%d_%H%M%S}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.automl, f)
        self.log(f"Model saved to {model_path}")
        
        # Log statistics
        self.log("\nModel Statistics:")
        self.log(self.automl.sprint_statistics())
        
        return self.automl
    
    def load_pretrained_model(self, model_path: str):
        """Load a pre-trained model."""
        self.log(f"Loading pre-trained model from {model_path}")
        
        with open(model_path, 'rb') as f:
            self.automl = pickle.load(f)
        
        self.log("Model loaded successfully")
        return self.automl
    
    def evaluate_model(self, X, y, split_name: str = "Test"):
        """Evaluate model on a dataset."""
        self.log(f"\n{'='*70}")
        self.log(f"Evaluating on {split_name} set...")
        self.log(f"{'='*70}")
        
        # Debug: Check feature alignment
        if hasattr(self.automl, 'feature_names_in_'):
            model_features = self.automl.feature_names_in_
            data_features = X.columns.tolist()
            
            self.log(f"\n🔍 Feature Alignment Check:")
            self.log(f"  Model expects {len(model_features)} features")
            self.log(f"  Data has {len(data_features)} features")
            
            missing_in_data = set(model_features) - set(data_features)
            extra_in_data = set(data_features) - set(model_features)
            
            if missing_in_data:
                self.log(f"  ⚠️ WARNING: {len(missing_in_data)} features MISSING in data:")
                self.log(f"     {sorted(list(missing_in_data))[:10]}")
            
            if extra_in_data:
                self.log(f"  ⚠️ WARNING: {len(extra_in_data)} EXTRA features in data:")
                self.log(f"     {sorted(list(extra_in_data))[:10]}")
            
            if not missing_in_data and not extra_in_data:
                self.log(f"  ✓ All features match!")
        
        # Get predictions
        probabilities = self.automl.predict_proba(X)[:, 1]
        predictions = self.automl.predict(X)
        
        # Compute metrics
        fpr, tpr, thresholds = roc_curve(y, probabilities, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y, probabilities)
        avg_precision = average_precision_score(y, probabilities)
        
        balanced_acc = balanced_accuracy_score(y, predictions)
        
        # Log metrics
        self.log(f"{split_name} Metrics:")
        self.log(f"  ROC AUC: {roc_auc:.4f}")
        self.log(f"  Average Precision: {avg_precision:.4f}")
        self.log(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Plot ROC and PR curves
        self._plot_roc_pr_curves(
            fpr, tpr, roc_auc, 
            recall, precision, avg_precision,
            split_name, y
        )
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y, probabilities, thresholds, split_name)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'true_label': y.values,
            'predicted_label': predictions,
            'probability_class_1': probabilities
        })
        pred_path = self.output_path / f'predictions_{split_name.lower()}_{self.feature_type}.csv'
        pred_df.to_csv(pred_path, index=False)
        self.log(f"Predictions saved to {pred_path}")
        
        return {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'balanced_accuracy': balanced_acc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
    
    def _plot_roc_pr_curves(self, fpr, tpr, roc_auc, recall, precision, 
                           avg_precision, split_name,y):
        """Plot ROC and Precision-Recall curves."""
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 12))
        
        # ROC Curve
        ax1.plot(fpr, tpr, lw=3, alpha=0.8, label=f'ROC AUC = {roc_auc:.2f}', color="#094268")
        ax1.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title(f'Test (lock-box) Set ROC Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax2.plot(recall, precision, lw=3, alpha=0.8, 
                label=f'Avg Precision = {avg_precision:.2f}', color="#094268")
        
        # Calculate baseline (proportion of positive class)
        baseline = (y == 1).sum() / len(y)  # Proportion of class 1
        ax2.plot([0, 1], [baseline, baseline], linestyle='--', lw=2, 
                color='#95a5a6', label='Random')
        
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title(f'Test (lock-box) Set Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_path / f'roc_pr_curves_{split_name.lower()}_{self.feature_type}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"ROC/PR curves saved to {plot_path}")
        
        # Save curve data
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        pr_df = pd.DataFrame({'recall': recall, 'precision': precision})
        roc_df.to_csv(self.output_path / f'roc_curve_{split_name.lower()}_{self.feature_type}.csv', index=False)
        pr_df.to_csv(self.output_path / f'pr_curve_{split_name.lower()}_{self.feature_type}.csv', index=False)
    
    def _plot_confusion_matrix(self, y_true, probabilities, thresholds, split_name):
        """Plot confusion matrix with optimal threshold."""
        # Find optimal threshold using Youden's J
        fpr, tpr, thresh = roc_curve(y_true, probabilities, pos_label=1)
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = thresh[optimal_idx]
        
        self.log(f"Optimal threshold (Youden's J): {optimal_threshold:.2f}")
        
        # Generate predictions
        y_pred = (probabilities >= optimal_threshold).astype(int)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=['FM+', 'FM−']
        )
        disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)
        ax.set_title(f'Test (lock-box) Set Confusion Matrix\n(threshold={optimal_threshold:.2f})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.plots_path / f'confusion_matrix_{split_name.lower()}_{self.feature_type}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"Confusion matrix saved to {plot_path}")
        
        # Log confusion matrix values
        self.log(f"\nConfusion Matrix:")
        self.log(f"  True Negatives:  {cm[0, 0]}")
        self.log(f"  False Positives: {cm[0, 1]}")
        self.log(f"  False Negatives: {cm[1, 0]}")
        self.log(f"  True Positives:  {cm[1, 1]}")
    
    def compute_feature_importance(self, X_test, y_test, n_repeats=100):
        """Compute and plot feature permutation importance with 100 repeats."""
        self.log("\n" + "="*70)
        self.log(f"Computing Feature Permutation Importance ({n_repeats} repeats)...")
        self.log("="*70)
        
        r = permutation_importance(
            self.automl,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
            scoring='roc_auc'
        )
        
        self.log(f"Completed {n_repeats} permutations for each of {len(X_test.columns)} features")
        
        # Sort by importance
        sort_idx = r.importances_mean.argsort()[::-1]
        
        # Create comprehensive plot showing ALL features (like your reference image)
        fig, ax = plt.subplots(figsize=(10, max(12, len(X_test.columns) * 0.3)))
        
        # Get sorted data
        sorted_features = [X_test.columns[i] for i in sort_idx]
        sorted_importances = r.importances[sort_idx]
        sorted_means = r.importances_mean[sort_idx]
        
        # Plot boxplots for all features
        positions = range(len(sorted_features))
        bp = ax.boxplot(
            sorted_importances.T,
            positions=positions,
            vert=False,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.6),
            medianprops=dict(color='darkblue', linewidth=2),
            whiskerprops=dict(color='gray', linewidth=1),
            capprops=dict(color='gray', linewidth=1)
        )
        
        # Add zero reference line
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=1)
        
        # Set labels
        ax.set_yticks(positions)
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel('Mean Importance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Features', fontsize=14, fontweight='bold')
        ax.set_title('Feature Permutation Importance', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to match your reference (most important at top)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plot_path = self.plots_path / f'feature_importance_{self.feature_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log(f"Feature importance plot saved to {plot_path}")
        
        # Save importance values with all statistics
        importance_df = pd.DataFrame({
            'feature': X_test.columns[sort_idx],
            'importance_mean': r.importances_mean[sort_idx],
            'importance_std': r.importances_std[sort_idx],
            'importance_min': r.importances[sort_idx].min(axis=1),
            'importance_25pct': np.percentile(r.importances[sort_idx], 25, axis=1),
            'importance_median': np.median(r.importances[sort_idx], axis=1),
            'importance_75pct': np.percentile(r.importances[sort_idx], 75, axis=1),
            'importance_max': r.importances[sort_idx].max(axis=1),
        })
        importance_path = self.output_path / f'feature_importance_{self.feature_type}.csv'
        importance_df.to_csv(importance_path, index=False)
        self.log(f"Feature importance values saved to {importance_path}")
        
        # Log top features
        self.log("\nTop 10 Most Important Features:")
        for i in range(min(10, len(sort_idx))):
            idx = sort_idx[i]
            self.log(f"  {i+1}. {X_test.columns[idx]}: "
                    f"{r.importances_mean[idx]:.4f} ± {r.importances_std[idx]:.4f}")
        
        # Log features with negative importance (if any)
        negative_importance = importance_df[importance_df['importance_mean'] < 0]
        if len(negative_importance) > 0:
            self.log(f"\n⚠️ {len(negative_importance)} features have negative mean importance:")
            for _, row in negative_importance.head(5).iterrows():
                self.log(f"  - {row['feature']}: {row['importance_mean']:.4f}")
        
        return r
    
    def run_permutation_test(self, X, y, n_permutations=100):
        """Run permutation test to assess statistical significance of model performance."""
        self.log("\n" + "="*70)
        self.log(f"Running Permutation Test ({n_permutations} permutations)...")
        self.log("="*70)
        
        # Get true performance
        y_pred_proba = self.automl.predict_proba(X)[:, 1]
        y_pred = self.automl.predict(X)
        
        true_roc_auc = auc(*roc_curve(y, y_pred_proba, pos_label=1)[:2])
        true_balanced_acc = balanced_accuracy_score(y, y_pred)
        true_avg_precision = average_precision_score(y, y_pred_proba)
        
        self.log(f"\nTrue Model Performance:")
        self.log(f"  ROC AUC: {true_roc_auc:.4f}")
        self.log(f"  Balanced Accuracy: {true_balanced_acc:.4f}")
        self.log(f"  Average Precision: {true_avg_precision:.4f}")
        
        # Run permutations
        self.log(f"\nRunning {n_permutations} permutations...")
        
        np.random.seed(42)  # For reproducibility
        
        perm_roc_aucs = []
        perm_balanced_accs = []
        perm_avg_precisions = []
        
        for i in range(n_permutations):
            if (i + 1) % 10 == 0:
                self.log(f"  Completed {i + 1}/{n_permutations} permutations...")
            
            # Shuffle labels
            y_permuted = y.sample(frac=1, random_state=i).reset_index(drop=True)
            
            # Get predictions (model stays the same, just labels are shuffled)
            y_perm_pred = self.automl.predict(X)
            y_perm_proba = self.automl.predict_proba(X)[:, 1]
            
            # Calculate metrics with permuted labels
            perm_roc_auc = auc(*roc_curve(y_permuted, y_perm_proba, pos_label=1)[:2])
            perm_balanced_acc = balanced_accuracy_score(y_permuted, y_perm_pred)
            perm_avg_precision = average_precision_score(y_permuted, y_perm_proba)
            
            perm_roc_aucs.append(perm_roc_auc)
            perm_balanced_accs.append(perm_balanced_acc)
            perm_avg_precisions.append(perm_avg_precision)
        
        # Calculate p-values
        p_value_roc = np.mean(np.array(perm_roc_aucs) >= true_roc_auc)
        p_value_balanced_acc = np.mean(np.array(perm_balanced_accs) >= true_balanced_acc)
        p_value_avg_precision = np.mean(np.array(perm_avg_precisions) >= true_avg_precision)
        
        self.log(f"\nPermutation Test Results:")
        self.log(f"  ROC AUC p-value: {p_value_roc:.4f}")
        self.log(f"  Balanced Accuracy p-value: {p_value_balanced_acc:.4f}")
        self.log(f"  Average Precision p-value: {p_value_avg_precision:.4f}")
        
        # Interpret results
        alpha = 0.05
        self.log(f"\nStatistical Significance (α = {alpha}):")
        self.log(f"  ROC AUC: {'SIGNIFICANT ✓' if p_value_roc < alpha else 'NOT SIGNIFICANT ✗'}")
        self.log(f"  Balanced Accuracy: {'SIGNIFICANT ✓' if p_value_balanced_acc < alpha else 'NOT SIGNIFICANT ✗'}")
        self.log(f"  Average Precision: {'SIGNIFICANT ✓' if p_value_avg_precision < alpha else 'NOT SIGNIFICANT ✗'}")
        
        # Plot permutation distributions
        self._plot_permutation_distributions(
            true_roc_auc, perm_roc_aucs, p_value_roc,
            true_balanced_acc, perm_balanced_accs, p_value_balanced_acc,
            true_avg_precision, perm_avg_precisions, p_value_avg_precision
        )
        
        # Save permutation results
        perm_results = pd.DataFrame({
            'permutation': range(n_permutations),
            'roc_auc': perm_roc_aucs,
            'balanced_accuracy': perm_balanced_accs,
            'avg_precision': perm_avg_precisions
        })
        
        perm_path = self.output_path / 'permutation_test_results.csv'
        perm_results.to_csv(perm_path, index=False)
        self.log(f"\nPermutation test results saved to {perm_path}")
        
        # Save summary
        summary = {
            'metric': ['ROC AUC', 'Balanced Accuracy', 'Average Precision'],
            'true_value': [true_roc_auc, true_balanced_acc, true_avg_precision],
            'perm_mean': [np.mean(perm_roc_aucs), np.mean(perm_balanced_accs), np.mean(perm_avg_precisions)],
            'perm_std': [np.std(perm_roc_aucs), np.std(perm_balanced_accs), np.std(perm_avg_precisions)],
            'p_value': [p_value_roc, p_value_balanced_acc, p_value_avg_precision],
            'significant': [p_value_roc < alpha, p_value_balanced_acc < alpha, p_value_avg_precision < alpha]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_path = self.output_path / 'permutation_test_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        self.log(f"Permutation test summary saved to {summary_path}")
        
        return {
            'p_values': {
                'roc_auc': p_value_roc,
                'balanced_accuracy': p_value_balanced_acc,
                'avg_precision': p_value_avg_precision
            },
            'permutation_distributions': {
                'roc_auc': perm_roc_aucs,
                'balanced_accuracy': perm_balanced_accs,
                'avg_precision': perm_avg_precisions
            }
        }
    
    def _plot_permutation_distributions(self, true_roc, perm_rocs, p_roc,
                                       true_bal_acc, perm_bal_accs, p_bal_acc,
                                       true_avg_prec, perm_avg_precs, p_avg_prec):
        """Plot permutation test distributions."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = [
            ('ROC AUC', true_roc, perm_rocs, p_roc),
            ('Balanced Accuracy', true_bal_acc, perm_bal_accs, p_bal_acc),
            ('Average Precision', true_avg_prec, perm_avg_precs, p_avg_prec)
        ]
        
        for ax, (metric_name, true_val, perm_vals, p_val) in zip(axes, metrics):
            # Plot histogram
            ax.hist(perm_vals, bins=30, alpha=0.7, color='#95a5a6', edgecolor='black')
            
            # Plot true value
            ax.axvline(true_val, color='#e74c3c', linewidth=3, linestyle='--', 
                      label=f'True Value = {true_val:.3f}')
            
            # Plot mean of permutations
            perm_mean = np.mean(perm_vals)
            ax.axvline(perm_mean, color='#3498db', linewidth=2, linestyle=':', 
                      label=f'Perm Mean = {perm_mean:.3f}')
            
            # Add text with p-value
            significance = '✓ Significant' if p_val < 0.05 else '✗ Not Significant'
            ax.text(0.02, 0.98, f'p = {p_val:.4f}\n{significance}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10, fontweight='bold')
            
            ax.set_xlabel(metric_name, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Permutation Test: {metric_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_path / 'permutation_test_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"Permutation test plots saved to {plot_path}")
    
    def run_training_pipeline(self):
        """Execute full training pipeline."""
        self.log("="*70)
        self.log("STARTING AUTOML TRAINING PIPELINE")
        self.log("="*70)
        
        # Load data
        train_ids, val_ids, test_ids, holdout_ids, excluded_ids, scores, features = self.load_data()
        
        # Prepare datasets
        datasets = self.prepare_datasets(
            train_ids, val_ids, test_ids, holdout_ids, excluded_ids, scores, features
        )
        
        # Train model
        self.train_model(datasets['X_train'], datasets['y_train'])
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(
            datasets['X_test'], 
            datasets['y_test'], 
            "Test"
        )
        
        # Evaluate on lockbox set
        lockbox_metrics = self.evaluate_model(
            datasets['X_lockbox'], 
            datasets['y_lockbox'], 
            "LockBox"
        )
        
        # Compute feature importance with 100 repeats
        self.compute_feature_importance(datasets['X_test'], datasets['y_test'], n_repeats=100)
        
        # Save reports
        self.save_data_loss_report()
        
        self.log("\n" + "="*70)
        self.log("TRAINING PIPELINE COMPLETED")
        self.log("="*70)
        self.log(f"All outputs saved to: {self.output_path}")
        
        return {
            'test_metrics': test_metrics,
            'lockbox_metrics': lockbox_metrics,
            'datasets': datasets
        }
    
    def run_inference_pipeline(self, model_path: str):
        """Execute inference pipeline with pre-trained model."""
        self.log("="*70)
        self.log("STARTING AUTOML INFERENCE PIPELINE")
        self.log("="*70)
        
        # Load pre-trained model
        self.load_pretrained_model(model_path)
        
        # Load data
        train_ids, val_ids, test_ids, holdout_ids, excluded_ids, scores, features = self.load_data()
        
        # Prepare datasets (this will apply --prereg mapping if enabled)
        datasets = self.prepare_datasets(
            train_ids, val_ids, test_ids, holdout_ids, excluded_ids, scores, features
        )
        

        # Evaluate on lockbox set
        lockbox_metrics = self.evaluate_model(
            datasets['X_lockbox'],
            datasets['y_lockbox'], 
            "LockBox"
        )
        
        # Compute feature permutation importance
        self.compute_feature_importance(
            datasets['X_lockbox'],
            datasets['y_lockbox'],
            n_repeats=100
        )
        
        # Save reports
        self.save_data_loss_report()
        
        self.log("\n" + "="*70)
        self.log("INFERENCE PIPELINE COMPLETED")
        self.log("="*70)
        self.log(f"All outputs saved to: {self.output_path}")
        
        return {
            'lockbox_metrics': lockbox_metrics,
            'datasets': datasets
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GMA AutoML Training/Inference')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], 
                       default='train', help='Training or inference mode')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--output-path', type=str, default='./automl_output',
                       help='Path for output files')
    parser.add_argument('--feature-type', type=str, choices=['total', 'windows'],
                       default='total', help='Feature type to use')
    parser.add_argument('--feature-file', type=str, default=None,
                       help='Path to feature CSV file (auto-detects from pose pipeline if not specified)')
    parser.add_argument('--no-exclusions', action='store_true',
                       help='Skip exclusions from all_excluded_videos.csv (include all videos with features/scores)')
    parser.add_argument('--prereg', action='store_true',
                       help='Apply pre-registered model feature name mapping (for inference with old models)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre-trained model (inference mode only)')
    parser.add_argument('--time-limit', type=int, default=300,
                       help='Total time limit for AutoML (seconds)')
    parser.add_argument('--per-run-limit', type=int, default=30,
                       help='Time limit per model run (seconds)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GMAAutoMLPipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        feature_type=args.feature_type,
        feature_file=args.feature_file,
        apply_exclusions=not args.no_exclusions,  # Invert the flag
        prereg=args.prereg,
        time_limit=args.time_limit,
        per_run_limit=args.per_run_limit
    )
    
    # Run appropriate pipeline
    if args.mode == 'train':
        results = pipeline.run_training_pipeline()
    else:
        if args.model_path is None:
            raise ValueError("--model-path required for inference mode")
        results = pipeline.run_inference_pipeline(args.model_path)
    
    print("\nPipeline completed successfully!")
    print(f"Check {args.output_path} for all outputs")