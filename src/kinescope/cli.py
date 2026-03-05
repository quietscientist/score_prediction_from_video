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


def _cmd_pretrain(args):
    from kinescope.pretrain.pretrain import pretrain
    pretrain(
        output_dir=args.output,
        datasets=args.datasets if args.datasets else None,
        data_dir=args.data_dir,
        data_root=args.data_root,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        artifacts_dir=args.artifacts_dir,
        device=args.device,
    )


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

    # ── pretrain ─────────────────────────────────────────────────────────────
    p_pt = sub.add_parser("pretrain", help="Pretrain Pose-JEPA ViT on COCO-17 pose data")
    p_pt.add_argument("--datasets",      nargs="+", metavar="DATASET",
                      choices=["amass", "ntu120", "humoto", "coco"],
                      help="Named datasets to load from $KINESCOPE_DATA_DIR "
                           "(amass, ntu120, humoto, coco). Mutually exclusive with --data-dir.")
    p_pt.add_argument("--data-dir",      dest="data_dir", default=None,
                      help="Explicit directory of COCO-17 CSV files or .npy clip arrays. "
                           "Use instead of --datasets for custom data.")
    p_pt.add_argument("--data-root",     dest="data_root", default=None,
                      help="Override $KINESCOPE_DATA_DIR for --datasets loading.")
    p_pt.add_argument("--output",        required=True, help="Output directory (checkpoints + metrics)")
    p_pt.add_argument("--epochs",        type=int,   default=100)
    p_pt.add_argument("--embed-dim",     type=int,   default=128,  dest="embed_dim")
    p_pt.add_argument("--n-layers",      type=int,   default=4,    dest="n_layers")
    p_pt.add_argument("--n-heads",       type=int,   default=4,    dest="n_heads")
    p_pt.add_argument("--seq-len",       type=int,   default=60,   dest="seq_len",
                      help="Frames per clip (must match data)")
    p_pt.add_argument("--batch-size",    type=int,   default=64,   dest="batch_size")
    p_pt.add_argument("--lr",            type=float, default=1e-4)
    p_pt.add_argument("--artifacts-dir", default="artifacts",     dest="artifacts_dir",
                      help="Directory for diagnostic visualizations (default: artifacts/)")
    p_pt.add_argument("--device",        default="auto",
                      choices=["auto", "cpu", "cuda", "mps"])
    p_pt.set_defaults(func=_cmd_pretrain)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
