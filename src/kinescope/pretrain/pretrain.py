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
from torch.utils.data import DataLoader, WeightedRandomSampler

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

    # Load clips
    if datasets is not None:
        from kinescope.pretrain.dataset_registry import load_clips
        clips = load_clips(datasets, seq_len=seq_len, data_root=data_root)
    elif data_dir is not None:
        data_dir = pathlib.Path(data_dir)
        npy_files = sorted(data_dir.glob("*.npy"))
        if npy_files:
            clips = np.concatenate([np.load(f) for f in npy_files], axis=0)
            print(f"Loaded {len(clips)} clips from {len(npy_files)} .npy files")
        else:
            print(f"Loading COCO-17 CSV files from {data_dir} ...")
            clips = _load_coco_clips(data_dir, seq_len=seq_len)
            print(f"Loaded {len(clips)} clips (seq_len={seq_len})")
    else:
        raise ValueError("Provide either datasets=['amass', ...] or data_dir='path/to/csvs'")

    if len(clips) == 0:
        raise ValueError(f"No clips found in {data_dir}. Supply .csv or .npy files.")

    # Motion-aware sampling
    motion_weights = ClipDataset.compute_motion_weights(clips)
    dataset = ClipDataset(clips, motion_weights)
    sampler = WeightedRandomSampler(
        torch.tensor(motion_weights), num_samples=len(dataset), replacement=True
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True
    )

    # Build Pose-JEPA model
    coord_dim = clips.shape[-1]  # 2 or 3
    model = PoseJEPA(
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
        coord_dim=coord_dim,
    ).to(dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

    jepa_losses = []
    tpc_losses = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_jepa, epoch_tpc = [], []
        t0 = time.time()

        for batch in loader:
            batch = batch.to(dev)
            out = model(batch)

            optimizer.zero_grad()
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()

            epoch_jepa.append(out["jepa_loss"].item())
            epoch_tpc.append(out["tpc_loss"].item())

        scheduler.step()

        mean_jepa = float(np.mean(epoch_jepa))
        mean_tpc = float(np.mean(epoch_tpc))
        jepa_losses.append(mean_jepa)
        tpc_losses.append(mean_tpc)

        print(
            f"Epoch {epoch:4d}/{epochs} | JEPA={mean_jepa:.4f} | TPC={mean_tpc:.4f}"
            f" | t={time.time() - t0:.1f}s"
        )

        total = mean_jepa + 0.5 * mean_tpc
        if total < best_loss:
            best_loss = total
            torch.save(
                {
                    "epoch": epoch,
                    "context_encoder": model.context_encoder.state_dict(),
                    "jepa_loss": mean_jepa,
                    "tpc_loss": mean_tpc,
                    "config": {
                        "embed_dim": embed_dim,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "seq_len": seq_len,
                        "coord_dim": coord_dim,
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
