"""
Pose-JEPA pretraining script.

Two usage modes:

1. Named datasets via $KINESCOPE_DATA_DIR:
    kinescope pretrain --datasets humoto amass --output ./checkpoints/ --epochs 100

   Datasets are loaded from $KINESCOPE_DATA_DIR (default: ~/kinescope_data/):
       ~/kinescope_data/amass/    → AMASS .npz files
       ~/kinescope_data/ntu120/   → NTU RGB+D 120 .skeleton files
       ~/kinescope_data/humoto/   → HUMOTO .fbx + .yaml files
       ~/kinescope_data/coco/     → Own COCO-17 .csv files

2. Explicit data directory (COCO-17 CSVs or .npy arrays):
    kinescope pretrain --data-dir ./pose_clips/ --output ./checkpoints/

Python API:
    from kinescope.pretrain.pretrain import pretrain
    metrics = pretrain(datasets=["humoto", "amass"], output_dir="./ckpt",
                       artifacts_dir="./artifacts")
"""

import json
import pathlib
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from kinescope.prediction._vit import PoseJEPA, _sample_block_mask
from kinescope.pretrain.clip_dataset import ClipDataset
from kinescope.pretrain.visualize import plot_loss_curve, plot_masked_clip, plot_pose_clip
from kinescope.skeleton import COCO_PART_NAMES


def _load_coco_clips(data_dir: pathlib.Path, seq_len: int = 60) -> np.ndarray:
    """
    Load COCO-17 CSV files, normalize skeletons, and chunk into fixed-length clips.

    Parameters
    ----------
    data_dir : Path — directory of COCO-17 CSV files
    seq_len : int — clip length in frames

    Returns
    -------
    (N, seq_len, 17, 2) float32 array
    """
    import pandas as pd

    from kinescope.pose.io import read_coco_csv
    from kinescope.processing.normalization import normalise_skeletons
    from kinescope.processing.smoothing import interpolate_df

    n_parts = len(COCO_PART_NAMES)
    all_clips = []
    csv_files = sorted(data_dir.glob("*.csv"))

    for csv_path in csv_files:
        try:
            df = read_coco_csv(csv_path)
            df = interpolate_df(df)
            df = normalise_skeletons(df)
        except Exception:
            continue

        frames = sorted(df["frame"].unique())
        T_total = len(frames)
        if T_total < seq_len:
            continue

        frame_to_idx = {f: i for i, f in enumerate(frames)}
        sequence = np.zeros((T_total, n_parts, 2), dtype=np.float32)

        part_index = {name: i for i, name in enumerate(COCO_PART_NAMES)}
        for _, row in df.iterrows():
            t = frame_to_idx[row["frame"]]
            j = part_index.get(row["bp"], -1)
            if j >= 0:
                sequence[t, j, 0] = row["x"] if not np.isnan(row["x"]) else 0.0
                sequence[t, j, 1] = row["y"] if not np.isnan(row["y"]) else 0.0

        step = seq_len // 2
        for start in range(0, T_total - seq_len + 1, step):
            all_clips.append(sequence[start : start + seq_len])

    if not all_clips:
        return np.empty((0, seq_len, n_parts, 2), dtype=np.float32)
    return np.stack(all_clips)


def pretrain(
    output_dir: str,
    datasets: Optional[list] = None,
    data_dir: Optional[str] = None,
    data_root: Optional[str] = None,
    epochs: int = 100,
    embed_dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    seq_len: int = 60,
    batch_size: int = 64,
    lr: float = 1e-4,
    artifacts_dir: Optional[str] = None,
    device: str = "auto",
    checkpoint_every: int = 10,
    mlflow_experiment: str = "pose-jepa-pretrain",
    use_amp: bool = True,
    max_clips: Optional[int] = None,
    tpc_weight: float = 0.0,
    invariant_weight: float = 0.0,
    long_horizon_weight: float = 0.0,
    long_horizon_segments: int = 4,
    ema_warmup_epochs: Optional[int] = None,
    sigreg_weight: float = 0.0,
    resume: Optional[str] = None,
    grad_clip: float = 5.0,
) -> dict:
    """
    Run Pose-JEPA pretraining.

    Parameters
    ----------
    output_dir : str — where to save checkpoints and metrics.json
    datasets : list of str or None — named datasets to load: ['amass', 'ntu120', 'humoto', 'coco']
        Loaded from $KINESCOPE_DATA_DIR / <name> (or data_root / <name>).
        If None, data_dir must be provided.
    data_dir : str or None — explicit directory of COCO-17 CSVs or .npy files.
        Used when datasets is None. Mutually exclusive with datasets.
    data_root : str or None — override $KINESCOPE_DATA_DIR for named dataset loading.
    epochs : int
    embed_dim : int — token embedding dimension (minimum 128)
    n_layers : int — transformer encoder layers
    n_heads : int — attention heads
    seq_len : int — frames per clip (must match data)
    batch_size : int
    lr : float — AdamW learning rate
    artifacts_dir : str or None — save visualizations here (PNG files)
    device : str — 'auto', 'cpu', 'cuda', 'mps'

    Returns
    -------
    dict with 'jepa_losses', 'tpc_losses', 'best_loss'
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Device: {dev}")
    _ = mlflow_experiment  # kept for CLI compatibility
    amp_enabled = use_amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    print(f"AMP: {'enabled' if amp_enabled else 'disabled'}")

    # Load clips
    if datasets is not None:
        from kinescope.pretrain.dataset_registry import load_clips
        clips = load_clips(datasets, seq_len=seq_len, data_root=data_root)
    elif data_dir is not None:
        data_dir = pathlib.Path(data_dir)
        npy_files = sorted(data_dir.glob("*.npy"))
        if npy_files:
            clips = np.concatenate([np.load(f, mmap_mode="r") for f in npy_files], axis=0)
            print(f"Loaded {len(clips)} clips from {len(npy_files)} .npy files")
            # Pre-sanitize in-place: eliminates per-item nan_to_num + copy overhead in __getitem__
            clips = np.nan_to_num(np.asarray(clips, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            print(f"Loading COCO-17 CSV files from {data_dir} ...")
            clips = _load_coco_clips(data_dir, seq_len=seq_len)
            print(f"Loaded {len(clips)} clips (seq_len={seq_len})")
    else:
        raise ValueError("Provide either datasets=['amass', ...] or data_dir='path/to/csvs'")

    if len(clips) == 0:
        source_desc = str(data_dir) if data_dir is not None else f"datasets={datasets}"
        raise ValueError(f"No clips found from {source_desc}. Supply .csv or .npy files.")

    if max_clips is not None and len(clips) > max_clips:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(clips), size=max_clips, replace=False)
        clips = np.asarray(clips[idx], dtype=np.float32)
        print(f"Subsampled to {len(clips)} clips (--max-clips)")

    # Motion-aware sampling — cache weights to avoid recomputing on 1.44M clips.
    # Use num_workers=0: for this small model, CUDA async execution overlaps data
    # loading with GPU compute, and IPC overhead from workers exceeds the benefit.
    # Use numpy for index generation (torch.multinomial is prohibitively slow at 1.44M).
    weights_cache = output_dir / "motion_weights.npy"
    if weights_cache.exists():
        print("Loading cached motion weights ...")
        motion_weights = np.load(weights_cache)
    else:
        print(f"Computing motion weights for {len(clips)} clips ...")
        motion_weights = ClipDataset.compute_motion_weights(clips)
        np.save(weights_cache, motion_weights)
        print(f"Motion weights cached to {weights_cache}")

    p = motion_weights / motion_weights.sum()
    sample_idx = np.random.choice(len(clips), size=len(clips), replace=True, p=p)

    from torch.utils.data import Sampler
    class _NumpySampler(Sampler):
        def __init__(self, idx): self.idx = idx
        def __iter__(self): return iter(self.idx.tolist())
        def __len__(self): return len(self.idx)

    dataset = ClipDataset(clips, motion_weights)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=_NumpySampler(sample_idx),
        num_workers=4, pin_memory=(dev.type == "cuda"), persistent_workers=True,
    )

    # Build Pose-JEPA model
    coord_dim = clips.shape[-1]  # 2 or 3
    model = PoseJEPA(
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
        coord_dim=coord_dim,
        tpc_weight=tpc_weight,
        invariant_weight=invariant_weight,
        long_horizon_weight=long_horizon_weight,
        long_horizon_segments=long_horizon_segments,
        sigreg_weight=sigreg_weight,
    ).to(dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    # eta_min=5% of peak LR prevents the optimizer from becoming ineffective late in training
    # while the EMA target encoder is still drifting. V-JEPA uses a cosine LR schedule
    # with a non-zero floor (Bardes et al., ICLR 2024, Appendix A).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    # Resume from checkpoint if requested
    start_epoch = 1
    jepa_losses = []
    tpc_losses = []
    invariant_losses = []
    long_horizon_losses = []
    sigreg_losses = []
    total_losses = []
    best_loss = float("inf")

    if resume:
        ckpt = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        # Fast-forward scheduler to match resumed epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        # Load existing metrics to continue appending
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                prior = json.load(f)
            jepa_losses = prior.get("jepa_losses", [])
            tpc_losses = prior.get("tpc_losses", [])
            invariant_losses = prior.get("invariant_losses", [])
            long_horizon_losses = prior.get("long_horizon_losses", [])
            sigreg_losses = prior.get("sigreg_losses", [])
            total_losses = prior.get("total_losses", [])
            best_loss = prior.get("best_loss", float("inf"))
        print(f"Resumed from {resume} — starting at epoch {start_epoch}, best_loss={best_loss:.4f}")

    # Visualize input data before training
    if artifacts_dir:
        artifacts_dir = pathlib.Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        sample_clip = clips[0]  # (T, 17, 2)
        plot_pose_clip(
            sample_clip,
            artifacts_dir / "synthetic_pose_clip.png",
            n_frames=9,
            title="Sample normalized COCO-17 clip (pretraining input)",
        )

        mask_t = _sample_block_mask(seq_len, len(COCO_PART_NAMES))
        plot_masked_clip(
            sample_clip,
            mask_t.numpy(),
            artifacts_dir / "jepa_mask_pattern.png",
            n_frames=9,
            title="Sample Pose-JEPA spatiotemporal block mask",
        )
        print(f"Input visualizations saved to {artifacts_dir}/")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_jepa, epoch_tpc, epoch_inv, epoch_lh, epoch_sig, epoch_total, epoch_grads = [], [], [], [], [], [], []
        t0 = time.time()

        skipped_steps = 0
        for batch in loader:
            batch = batch.to(dev)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(batch)

            loss = out["total_loss"]

            # Warn loudly if loss is non-finite before backward
            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss ({loss.item():.4f}) at epoch {epoch} — skipping step")
                skipped_steps += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

            # Detect non-finite gradients explicitly
            if not np.isfinite(grad_norm):
                print(f"  WARNING: non-finite grad_norm at epoch {epoch} — AMP will skip step")

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # Only update EMA when optimizer step was actually taken
            if scaler.get_scale() == scale_before:
                model.update_ema()
            else:
                skipped_steps += 1

            epoch_jepa.append(out["jepa_loss"].item())
            epoch_tpc.append(out["tpc_loss"].item())
            epoch_inv.append(out["invariant_loss"].item())
            epoch_lh.append(out["long_horizon_loss"].item())
            epoch_sig.append(out["sigreg_loss"].item())
            epoch_total.append(out["total_loss"].item())
            epoch_grads.append(grad_norm)

        if skipped_steps > 0:
            print(f"  WARNING: {skipped_steps} steps skipped this epoch (inf/nan gradients)")

        scheduler.step()
        model.step_ema_decay(epoch, epochs, warmup_epochs=ema_warmup_epochs)

        mean_jepa = float(np.mean(epoch_jepa))
        mean_tpc = float(np.mean(epoch_tpc))
        mean_inv = float(np.mean(epoch_inv))
        mean_lh = float(np.mean(epoch_lh))
        mean_sig = float(np.mean(epoch_sig))
        mean_total = float(np.mean(epoch_total))
        finite_grads = [g for g in epoch_grads if np.isfinite(g)]
        mean_grad = float(np.mean(finite_grads)) if finite_grads else float("nan")
        current_lr = scheduler.get_last_lr()[0]
        jepa_losses.append(mean_jepa)
        tpc_losses.append(mean_tpc)
        invariant_losses.append(mean_inv)
        long_horizon_losses.append(mean_lh)
        sigreg_losses.append(mean_sig)
        total_losses.append(mean_total)

        tpc_active = out["tpc_active_fraction"].item()
        sig_str = f" | SIGReg={mean_sig:.4f}" if sigreg_weight > 0 else ""
        print(
            f"Epoch {epoch:4d}/{epochs} | JEPA={mean_jepa:.4f} | TPC={mean_tpc:.4f}"
            f" | TPC_active={tpc_active:.2f}{sig_str} | grad={mean_grad:.3f} | lr={current_lr:.2e}"
            f" | ema_τ={model.current_ema_decay:.4f} | t={time.time() - t0:.1f}s"
        )

        if epoch % checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "context_encoder":  model.context_encoder.state_dict(),
                    "target_encoder":   model.target_encoder.state_dict(),
                    "predictor":        model.predictor.state_dict(),
                },
                output_dir / f"checkpoint_ep{epoch:04d}.pt",
            )
            print(f"  → Checkpoint saved: checkpoint_ep{epoch:04d}.pt")

        if mean_total < best_loss:
            best_loss = mean_total
            torch.save(
                {
                    "epoch": epoch,
                    "context_encoder": model.context_encoder.state_dict(),
                    "jepa_loss": mean_jepa,
                    "tpc_loss": mean_tpc,
                    "invariant_loss": mean_inv,
                    "total_loss": mean_total,
                    "config": {
                        "embed_dim": embed_dim,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "seq_len": seq_len,
                        "coord_dim": coord_dim,
                        "tpc_weight": tpc_weight,
                        "invariant_weight": invariant_weight,
                        "long_horizon_weight": long_horizon_weight,
                        "long_horizon_segments": long_horizon_segments,
                        "ema_warmup_epochs": ema_warmup_epochs,
                        "sigreg_weight": sigreg_weight,
                    },
                },
                output_dir / "best.pt",
            )

    # Save latest checkpoint
    torch.save(
        {
            "epoch": epochs,
            "context_encoder": model.context_encoder.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        output_dir / "latest.pt",
    )

    metrics = {
        "jepa_losses": jepa_losses,
        "tpc_losses": tpc_losses,
        "invariant_losses": invariant_losses,
        "long_horizon_losses": long_horizon_losses,
        "sigreg_losses": sigreg_losses,
        "total_losses": total_losses,
        "best_loss": best_loss,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if artifacts_dir:
        plot_loss_curve(
            jepa_losses,
            tpc_losses,
            pathlib.Path(artifacts_dir) / "pretrain_loss_curve.png",
        )
        print(f"Loss curve saved to {artifacts_dir}/pretrain_loss_curve.png")

    print(f"Done. Best checkpoint: {output_dir}/best.pt  (loss={best_loss:.4f})")
    return metrics
