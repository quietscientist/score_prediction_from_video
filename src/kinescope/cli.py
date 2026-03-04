"""
kinescope command-line interface.

Usage:
  kinescope process --input <pose_dir> --output <out_dir> --dataset <name> --vid-info <csv>
  kinescope predict --features <csv> --scores <csv> --output <out_dir> [--model xgboost]
  kinescope convert --input <file> --format <fmt> --output <file>
"""

import argparse
import sys


def _cmd_process(args):
    from kinescope.pipeline.pose_pipeline import PoseProcessingPipeline
    pipeline = PoseProcessingPipeline(
        dataset=args.dataset,
        pose_dir=args.input,
        vid_info_csv=args.vid_info,
        output_path=args.output,
        n_workers=args.workers,
        overwrite=not args.no_overwrite,
    )
    pipeline.run()


def _cmd_predict(args):
    from kinescope.prediction.train import train_and_evaluate, load_features_and_labels
    X, y = load_features_and_labels(
        features_csv=args.features,
        scores_csv=args.scores,
    )
    train_and_evaluate(
        X, y,
        model_name=args.model,
        splits_dir=args.splits_dir,
        output_dir=args.output,
        binary=not args.multiclass,
    )


def _cmd_convert(args):
    from kinescope.pose.converter import convert_to_coco
    df = convert_to_coco(args.input, fmt=args.format, fps=args.fps)
    df.to_csv(args.output, index=False)
    print(f"Converted {len(df)} keypoint rows → {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="kinescope",
        description="Predict scores from human movement video (COCO-17 pose format).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── process ──────────────────────────────────────────────────────────────
    p_proc = sub.add_parser("process", help="Extract kinematic features from COCO-17 pose CSVs")
    p_proc.add_argument("--input",      required=True, help="Directory of COCO-17 pose CSV files")
    p_proc.add_argument("--output",     required=True, help="Output directory")
    p_proc.add_argument("--dataset",    required=True, help="Dataset name (used in output naming)")
    p_proc.add_argument("--vid-info",   required=True, dest="vid_info",
                        help="CSV with columns: video, fps, width, height")
    p_proc.add_argument("--workers",    type=int, default=8)
    p_proc.add_argument("--no-overwrite", action="store_true")
    p_proc.set_defaults(func=_cmd_process)

    # ── predict ──────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Train and evaluate a score prediction model")
    p_pred.add_argument("--features",   required=True, help="Consolidated feature CSV")
    p_pred.add_argument("--scores",     required=True, help="Scores CSV (video, score columns)")
    p_pred.add_argument("--output",     required=True, help="Output directory for plots/model")
    p_pred.add_argument("--model",      default="xgboost",
                        choices=["logreg", "xgboost", "random_forest", "vit"])
    p_pred.add_argument("--splits-dir", dest="splits_dir", default=None,
                        help="Directory with train.csv / test.csv split files")
    p_pred.add_argument("--multiclass", action="store_true",
                        help="Keep all score classes (default: binary classification)")
    p_pred.set_defaults(func=_cmd_predict)

    # ── convert ──────────────────────────────────────────────────────────────
    p_conv = sub.add_parser("convert", help="Convert pose data from other formats to COCO-17 CSV")
    p_conv.add_argument("--input",   required=True, help="Input pose file")
    p_conv.add_argument("--format",  required=True,
                        choices=["coco", "openpose", "mediapipe", "kinect_v2", "smpl"])
    p_conv.add_argument("--output",  required=True, help="Output COCO-17 CSV path")
    p_conv.add_argument("--fps",     type=float, default=None, help="Frames per second (optional)")
    p_conv.set_defaults(func=_cmd_convert)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
