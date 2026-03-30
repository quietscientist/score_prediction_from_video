#!/usr/bin/env python3
"""Audit cached pretraining clips using canonical kinescope kinematics modules."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kinescope.kinematics import (
    angle_features,
    corr_lr,
    get_dynamics_angle,
    get_dynamics_xy,
    get_joint_angles,
    xy_features,
)
from kinescope.pose.io import add_neck
from kinescope.skeleton import COCO_PART_NAMES


_BP_NAMES = np.array(COCO_PART_NAMES)


def _clip_to_df(clip: np.ndarray, video_id: str, fps: float) -> pd.DataFrame:
    t, j, d = clip.shape
    if d != 2:
        raise ValueError(f"Expected 2D clip, got last dim={d}")
    frame_idx = np.repeat(np.arange(t), j)
    part_idx = np.tile(np.arange(j), t)
    rows = pd.DataFrame(
        {
            "video": video_id,
            "frame": frame_idx,
            "bp": _BP_NAMES[part_idx],
            "part_idx": part_idx,
            "x": clip[:, :, 0].reshape(-1),
            "y": clip[:, :, 1].reshape(-1),
            "confidence": 1.0,
            "fps": fps,
        }
    )
    rows["delta_t"] = 1.0 / fps
    rows["time"] = rows["frame"] / fps
    rows["pixel_x"] = 1.0
    rows["pixel_y"] = 1.0
    return rows


def _compute_clip_metrics(clip: np.ndarray, video_id: str, fps: float, delta_window: float) -> dict[str, float]:
    clip = np.nan_to_num(clip.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    df = _clip_to_df(clip, video_id=video_id, fps=fps)

    # XY dynamics and feature aggregates
    xdf = get_dynamics_xy(df, delta_window=delta_window).replace([np.inf, -np.inf], np.nan)
    xyf = xdf.groupby(["bp", "video"]).apply(xy_features).reset_index(drop=True)

    lr_corr = []
    for part in ("wrist", "elbow", "knee", "ankle"):
        sub = xdf[xdf["part"] == part]
        if {"left", "right"}.issubset(set(sub["side"].unique())):
            c = corr_lr(sub, "speed")
            if pd.notna(c):
                lr_corr.append(float(c))

    # Angle dynamics and feature aggregates
    adf = get_joint_angles(add_neck(df)).replace([np.inf, -np.inf], np.nan)
    if len(adf) > 0:
        adf = get_dynamics_angle(adf, delta_window=delta_window).replace([np.inf, -np.inf], np.nan)
        af = adf.groupby(["bp", "video"]).apply(angle_features).reset_index(drop=True)
        ang_vel = float(af["median_vel_angle"].median())
        ang_acc_iqr = float(af["IQR_acc_angle"].mean())
    else:
        ang_vel = np.nan
        ang_acc_iqr = np.nan

    return {
        "coord_max_abs": float(np.abs(clip).max()),
        "speed_mean": float(xdf["speed"].abs().median()),
        "accel_mean": float(np.nanmedian(np.sqrt(xdf["acceleration_x"] ** 2 + xdf["acceleration_y"] ** 2))),
        "xy_entropy_mean": float(xyf["meanent"].mean()),
        "xy_acc_iqr_mean": float(np.nanmean((xyf["IQRaccx"] + xyf["IQRaccy"]) / 2.0)),
        "angle_vel_median": ang_vel,
        "angle_acc_iqr_mean": ang_acc_iqr,
        "lr_speed_corr_mean": float(np.mean(lr_corr)) if lr_corr else np.nan,
    }


def _summarize(name: str, metrics: dict[str, np.ndarray], n_total: int) -> dict[str, float]:
    row: dict[str, float] = {
        "dataset": name,
        "n_clips_total": int(n_total),
        "n_clips_sampled": int(len(next(iter(metrics.values())))),
    }
    for k, v in metrics.items():
        vals = v[np.isfinite(v)]
        if len(vals) == 0:
            row[f"{k}_mean"] = np.nan
            row[f"{k}_std"] = np.nan
            row[f"{k}_p50"] = np.nan
            row[f"{k}_p95"] = np.nan
            continue
        row[f"{k}_mean"] = float(np.mean(vals))
        row[f"{k}_std"] = float(np.std(vals))
        row[f"{k}_p50"] = float(np.quantile(vals, 0.50))
        row[f"{k}_p95"] = float(np.quantile(vals, 0.95))
    return row


def _plot_hist(all_metrics: dict[str, dict[str, np.ndarray]], out_path: pathlib.Path) -> None:
    metric_names = [
        "coord_max_abs",
        "speed_mean",
        "accel_mean",
        "xy_entropy_mean",
        "angle_vel_median",
        "lr_speed_corr_mean",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_names):
        for dataset, m in all_metrics.items():
            vals = m[metric]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=80, alpha=0.35, density=True, label=dataset)
        ax.set_title(metric)
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached pretraining clip distributions.")
    parser.add_argument("--cache-dir", required=True, help="Directory with cached *.npy clip files")
    parser.add_argument("--output-dir", default="artifacts/data_audit", help="Where to save audit outputs")
    parser.add_argument(
        "--max-sample-per-dataset",
        type=int,
        default=50000,
        help="Max clips to sample per dataset for distribution plots",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used for dynamics calculations")
    parser.add_argument("--delta-window", type=float, default=0.3, help="Smoothing window (seconds) for kinematic dynamics")
    args = parser.parse_args()

    cache_dir = pathlib.Path(args.cache_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(cache_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {cache_dir}")

    rng = np.random.default_rng(args.seed)
    all_metrics: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, float]] = []

    for npy_path in npy_files:
        name = npy_path.stem
        arr = np.load(npy_path, mmap_mode="r")
        n_total = len(arr)
        if n_total == 0:
            continue

        n_take = min(n_total, args.max_sample_per_dataset)
        idx = rng.choice(n_total, size=n_take, replace=False) if n_take < n_total else np.arange(n_total)
        clips = np.asarray(arr[idx], dtype=np.float32)
        metrics_rows = []
        for i, clip in enumerate(clips):
            metrics_rows.append(
                _compute_clip_metrics(
                    clip,
                    video_id=f"{name}_{i}",
                    fps=args.fps,
                    delta_window=args.delta_window,
                )
            )
        metrics_df = pd.DataFrame(metrics_rows)
        metrics = {c: metrics_df[c].to_numpy(dtype=np.float32) for c in metrics_df.columns}
        all_metrics[name] = metrics
        summary_rows.append(_summarize(name, metrics, n_total))
        print(f"{name}: sampled {n_take}/{n_total} clips")

    if not all_metrics:
        raise RuntimeError("No non-empty datasets found in cache directory.")

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset")
    summary_csv = out_dir / "audit_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    hist_png = out_dir / "audit_metrics_hist.png"
    _plot_hist(all_metrics, hist_png)

    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved plots:   {hist_png}")


if __name__ == "__main__":
    main()
