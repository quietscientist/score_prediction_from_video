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


def _cmd_linearprobe(args):
    from kinescope.linearprobe.evaluate import run_linear_probe
    run_linear_probe(
        data_dir=args.data_dir,
        pretrained_weights=args.pretrained_weights,
        output_dir=args.output,
        ridge_alpha=args.ridge_alpha,
        device=args.device,
        artifacts_dir=args.artifacts_dir,
    )


def _cmd_cache_datasets(args):
    import pathlib
    import numpy as np
    from kinescope.pretrain.dataset_registry import load_clips

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        out_path = out / f"{name}.npy"
        if out_path.exists() and not args.overwrite:
            print(f"  {name}: {out_path} already exists, skipping (--overwrite to force)")
            continue
        print(f"Caching {name} ...")
        clips = load_clips([name], seq_len=args.seq_len, data_root=args.data_root, verbose=True)
        np.save(out_path, clips)
        print(f"  → saved {len(clips)} clips to {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")


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
        checkpoint_every=args.checkpoint_every,
        mlflow_experiment=args.mlflow_experiment,
        use_amp=not args.no_amp,
        max_clips=args.max_clips,
        tpc_weight=args.tpc_weight,
        invariant_weight=args.invariant_weight,
        long_horizon_weight=args.long_horizon_weight,
        long_horizon_segments=args.long_horizon_segments,
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

    # ── cache-datasets ───────────────────────────────────────────────────────
    p_cache = sub.add_parser("cache-datasets",
                             help="Pre-process datasets and save as .npy for fast future loading")
    p_cache.add_argument("--datasets",  nargs="+", required=True, metavar="DATASET",
                         choices=["amass", "ntu120", "humoto", "penn", "coco"],
                         help="Datasets to cache")
    p_cache.add_argument("--output",    required=True,
                         help="Directory to write <dataset>.npy files")
    p_cache.add_argument("--seq-len",   type=int, default=60, dest="seq_len")
    p_cache.add_argument("--data-root", default=None, dest="data_root",
                         help="Override $KINESCOPE_DATA_DIR")
    p_cache.add_argument("--overwrite", action="store_true",
                         help="Re-cache even if output file already exists")
    p_cache.set_defaults(func=_cmd_cache_datasets)

    # ── pretrain ─────────────────────────────────────────────────────────────
    p_pt = sub.add_parser("pretrain", help="Pretrain Pose-JEPA ViT on COCO-17 pose data")
    p_pt.add_argument("--datasets",      nargs="+", metavar="DATASET",
                      choices=["amass", "ntu120", "humoto", "penn", "coco"],
                      help="Named datasets to load from $KINESCOPE_DATA_DIR "
                           "(amass, ntu120, humoto, penn, coco). Mutually exclusive with --data-dir.")
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
    p_pt.add_argument("--device",              default="auto",
                      choices=["auto", "cpu", "cuda", "mps"])
    p_pt.add_argument("--checkpoint-every",   type=int, default=10, dest="checkpoint_every",
                      help="Save periodic checkpoint every N epochs (default: 10)")
    p_pt.add_argument("--mlflow-experiment",  default="pose-jepa-pretrain", dest="mlflow_experiment",
                      help="MLflow experiment name (default: pose-jepa-pretrain)")
    p_pt.add_argument("--no-amp", action="store_true", dest="no_amp",
                      help="Disable automatic mixed precision (AMP). Use if AMP causes NaN losses.")
    p_pt.add_argument("--max-clips", type=int, default=None, dest="max_clips",
                      help="Cap the number of clips loaded (random subset). Useful for smoke tests.")
    p_pt.add_argument("--tpc-weight", type=float, default=0.0, dest="tpc_weight",
                      help="Weight for TPC auxiliary loss (default: 0.0 = JEPA only). "
                           "Set >0 to re-enable TPC (see Option C in plan for better approach).")
    p_pt.add_argument("--invariant-weight", type=float, default=0.0, dest="invariant_weight",
                      help="Weight for clip-level kinematic invariant auxiliary loss "
                           "(symmetry/smoothness/coordination/entropy).")
    p_pt.add_argument("--long-horizon-weight", type=float, default=0.0, dest="long_horizon_weight",
                      help="Weight for coarse segment-level future latent prediction objective.")
    p_pt.add_argument("--long-horizon-segments", type=int, default=4, dest="long_horizon_segments",
                      help="Number of temporal segments per clip for long-horizon objective (default: 4).")
    p_pt.set_defaults(func=_cmd_pretrain)

    # ── probe (linearprobe) ───────────────────────────────────────────────────
    # Architecture config is read automatically from the checkpoint file.
    # For a random-init baseline, omit --pretrained-weights (defaults to 128/4/4/60).
    for probe_name in ("probe", "linearprobe"):
        p_lp = sub.add_parser(probe_name,
                               help="LOSO linear probe evaluation on UDysRS (alias: probe / linearprobe)")
        p_lp.add_argument("--data-dir",           required=True, dest="data_dir",
                          help="Path to UDysRS_UPDRS_Export directory")
        p_lp.add_argument("--pretrained-weights", default=None, dest="pretrained_weights",
                          help="Path to pretrained ViT checkpoint (.pt). "
                               "Architecture config is read from the checkpoint. "
                               "Omit for random-init baseline (128/4/4).")
        p_lp.add_argument("--output",             default="results/linearprobe",
                          help="Output directory for metrics and plots (default: results/linearprobe)")
        p_lp.add_argument("--ridge-alpha",        type=float, default=1.0,  dest="ridge_alpha")
        p_lp.add_argument("--device",             default="auto",
                          choices=["auto", "cpu", "cuda", "mps"])
        p_lp.add_argument("--artifacts-dir",      default="artifacts",      dest="artifacts_dir")
        p_lp.set_defaults(func=_cmd_linearprobe)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
