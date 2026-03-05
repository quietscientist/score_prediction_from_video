"""Pose-JEPA ViT for clinical and sports score prediction from COCO-17 pose timeseries.

Architecture:
    - PoseTokenEmbedding: per-joint coordinate → embed_dim with joint-type + temporal PE
    - PoseViT: transformer encoder (shared between context and target encoders)
    - JEPAPredictor: small transformer predicting target embeddings from context
    - TPCDecoder: predicts future pose coordinates from past context
    - PoseViTClassifier: sklearn-compatible wrapper for downstream score prediction

Pretraining objectives (see kinescope.pretrain.pretrain):
    1. Pose-JEPA (primary): predict target encoder embeddings for masked spatiotemporal blocks.
       No collapse risk — EMA target encoder naturally avoids trivial solutions.
    2. Motion-gated TPC (auxiliary): predict second half of clip from first half,
       only when mean joint displacement > motion_threshold. Avoids "predict same pose"
       shortcut during downtime.

References:
    - V-JEPA: Bardes et al., 2024 (Meta AI) — adapted to skeleton sequences
    - MotionBERT: Zhu et al., ICCV 2023 — inspiration for per-joint tokenization
"""

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from kinescope.skeleton import COCO_PART_INDEX, COCO_PART_NAMES, LIMB_JOINTS

N_JOINTS = len(COCO_PART_NAMES)  # 17
LIMB_JOINT_INDICES = [COCO_PART_INDEX[j] for j in LIMB_JOINTS]


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for temporal positions."""

    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, t_indices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t_indices : (N,) int tensor — frame indices

        Returns
        -------
        (N, embed_dim) positional encoding
        """
        return self.pe[t_indices]


class PoseTokenEmbedding(nn.Module):
    """
    Embeds COCO-17 joint coordinates into per-joint token embeddings.

    Each (frame, joint) pair becomes one token:
        [x, y] → linear projection + joint-type embedding + temporal PE

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension (minimum 128 recommended)
    n_joints : int
        Number of keypoints (17 for COCO-17)
    coord_dim : int
        Coordinate dimensions: 2 for (x,y) or 3 for (x,y,z)
    max_seq_len : int
        Maximum number of frames
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_joints: int = N_JOINTS,
        coord_dim: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_joints = n_joints

        self.coord_proj = nn.Linear(coord_dim, embed_dim)
        self.joint_embedding = nn.Embedding(n_joints, embed_dim)
        self.temporal_pe = SinusoidalPE(embed_dim, max_len=max_seq_len)

        # Up-weight initialization for clinically-informative limb joints
        with torch.no_grad():
            for idx in LIMB_JOINT_INDICES:
                self.joint_embedding.weight[idx] *= 1.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, J, D) float tensor — normalized pose sequences

        Returns
        -------
        tokens : (B, T*J, embed_dim) — flattened joint tokens
        """
        B, T, J, D = x.shape
        t_indices = torch.arange(T, device=x.device)
        j_indices = torch.arange(J, device=x.device)

        t_pe = self.temporal_pe(t_indices)      # (T, E)
        j_emb = self.joint_embedding(j_indices)  # (J, E)
        coord_emb = self.coord_proj(x)           # (B, T, J, E)

        # Add temporal PE (over B, J) and joint embedding (over B, T)
        tokens = coord_emb + t_pe.unsqueeze(1) + j_emb.unsqueeze(0)  # (B, T, J, E)
        return tokens.view(B, T * J, self.embed_dim)

    def position_encoding(
        self,
        t_indices: torch.Tensor,
        j_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Positional encoding for arbitrary (frame, joint) index pairs.
        Used by the JEPA predictor to encode masked token positions.

        Parameters
        ----------
        t_indices : (N,) int tensor — frame indices
        j_indices : (N,) int tensor — joint indices

        Returns
        -------
        (N, embed_dim) positional encodings
        """
        t_pe = self.temporal_pe(t_indices)       # (N, E)
        j_emb = self.joint_embedding(j_indices)  # (N, E)
        return t_pe + j_emb


class PoseViT(nn.Module):
    """
    Transformer encoder for COCO-17 pose sequences.

    Used as both context encoder and (EMA copy as) target encoder in Pose-JEPA,
    and as the fine-tuning backbone for score prediction.

    Parameters
    ----------
    embed_dim : int
    n_layers : int
    n_heads : int
    seq_len : int — frames per clip (sets max temporal PE length)
    n_joints : int
    coord_dim : int
    dropout : float
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 60,
        n_joints: int = N_JOINTS,
        coord_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.n_joints = n_joints

        self.token_embedding = PoseTokenEmbedding(
            embed_dim, n_joints, coord_dim, max_seq_len=seq_len + 64
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # [CLS] token for sequence-level representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, J, D) float tensor — normalized pose sequences

        Returns
        -------
        (B, embed_dim) — [CLS] token output (sequence-level representation)
        """
        return self.encode_tokens(x)[:, 0]

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return all token embeddings including [CLS].
        Used by Pose-JEPA context/target encoders.

        Returns
        -------
        (B, T*J+1, embed_dim)
        """
        B = x.shape[0]
        tokens = self.token_embedding(x)              # (B, T*J, E)
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, E)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, T*J+1, E)
        out = self.transformer(tokens)
        return self.norm(out)


class JEPAPredictor(nn.Module):
    """
    Pose-JEPA predictor: predicts target encoder embeddings for masked tokens.

    Takes context encoder output and the positional encodings of masked tokens,
    then predicts what the target encoder would output for those masked positions.

    Parameters
    ----------
    embed_dim : int — must match PoseViT embed_dim
    n_heads : int
    n_layers : int — smaller than the context encoder (2 is usually sufficient)
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        masked_pos_encodings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context_embeddings : (B, N_ctx, embed_dim) — encoded unmasked tokens
        masked_pos_encodings : (B, N_mask, embed_dim) — positional encodings of masked positions

        Returns
        -------
        (B, N_mask, embed_dim) — predicted embeddings for masked positions
        """
        out = self.transformer(masked_pos_encodings, context_embeddings)
        return self.norm(out)


class TPCDecoder(nn.Module):
    """
    Temporal Predictive Coding decoder.

    Given the [CLS] embedding from the first half of a clip, predicts
    raw (x,y) coordinates for the second half.

    Parameters
    ----------
    embed_dim : int
    n_joints : int
    coord_dim : int
    half_seq_len : int — number of frames to predict (= seq_len // 2)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_joints: int = N_JOINTS,
        coord_dim: int = 2,
        half_seq_len: int = 30,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.coord_dim = coord_dim
        self.half_seq_len = half_seq_len

        out_dim = half_seq_len * n_joints * coord_dim
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, out_dim),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cls_embedding : (B, embed_dim) — from context encoder on first half of clip

        Returns
        -------
        (B, half_seq_len, n_joints, coord_dim)
        """
        B = cls_embedding.shape[0]
        return self.decoder(cls_embedding).view(
            B, self.half_seq_len, self.n_joints, self.coord_dim
        )


def _sample_block_mask(
    T: int,
    J: int,
    mask_ratio: float = 0.5,
    device: Optional[torch.device] = None,
    limb_weight: float = 2.0,
) -> torch.Tensor:
    """
    Sample a spatiotemporal block mask for Pose-JEPA.

    Masks a contiguous temporal block of a random subset of joints.

    Parameters
    ----------
    T : int — number of frames
    J : int — number of joints
    mask_ratio : float — approximate fraction of tokens to mask
    device : torch.device or None
    limb_weight : float — relative sampling weight for limb joints

    Returns
    -------
    (T, J) bool tensor — True = masked
    """
    mask = torch.zeros(T, J, dtype=torch.bool, device=device)

    # Random contiguous temporal block
    block_len = int(T * (0.33 + torch.rand(1).item() * 0.17))
    block_len = max(1, min(block_len, T - 1))
    t_start = torch.randint(0, max(1, T - block_len), (1,)).item()

    # Joint sampling weights (limb joints more likely to be masked)
    weights = torch.ones(J, device=device)
    for idx in LIMB_JOINT_INDICES:
        weights[idx] = limb_weight
    weights = weights / weights.sum()

    n_joints_to_mask = max(1, int(mask_ratio * J))
    n_joints_to_mask = min(n_joints_to_mask, J)
    joint_indices = torch.multinomial(weights, n_joints_to_mask, replacement=False)

    mask[t_start : t_start + block_len, joint_indices] = True
    return mask


class PoseJEPA(nn.Module):
    """
    Full Pose-JEPA pretraining model.

    Includes context encoder, EMA target encoder, JEPA predictor, and TPC decoder.
    Call update_ema() after each optimizer step.

    Only context_encoder weights are saved/loaded for downstream fine-tuning.

    Parameters
    ----------
    embed_dim : int
    n_layers : int
    n_heads : int
    seq_len : int
    n_joints : int
    coord_dim : int
    ema_decay : float — τ for EMA update (default 0.996 per V-JEPA)
    motion_threshold : float — min mean joint displacement for TPC gate
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 60,
        n_joints: int = N_JOINTS,
        coord_dim: int = 2,
        ema_decay: float = 0.996,
        motion_threshold: float = 0.05,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_joints = n_joints
        self.coord_dim = coord_dim
        self.ema_decay = ema_decay
        self.motion_threshold = motion_threshold

        self.context_encoder = PoseViT(
            embed_dim, n_layers, n_heads, seq_len, n_joints, coord_dim
        )

        # Target encoder: EMA copy of context encoder (no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = JEPAPredictor(embed_dim, n_heads, n_layers=2)
        self.tpc_decoder = TPCDecoder(
            embed_dim, n_joints, coord_dim, half_seq_len=seq_len // 2
        )

    @torch.no_grad()
    def update_ema(self):
        """EMA update of target encoder. Call after each optimizer step."""
        τ = self.ema_decay
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            tgt_p.data.mul_(τ).add_(ctx_p.data, alpha=1 - τ)

    def _motion_gate(self, x_first_half: torch.Tensor) -> torch.Tensor:
        """
        Returns (B,) bool tensor: True where clip has significant motion.
        """
        displacements = (x_first_half[:, 1:] - x_first_half[:, :-1]).norm(dim=-1)
        return displacements.mean(dim=(1, 2)) > self.motion_threshold

    def forward(self, x: torch.Tensor) -> dict:
        """
        Compute Pose-JEPA + motion-gated TPC losses.

        Parameters
        ----------
        x : (B, T, J, D) float tensor — normalized pose sequences

        Returns
        -------
        dict: jepa_loss, tpc_loss, total_loss, tpc_active_fraction
        """
        B, T, J, D = x.shape
        device = x.device

        # --- Pose-JEPA ---
        masks = torch.stack(
            [_sample_block_mask(T, J, mask_ratio=0.5, device=device) for _ in range(B)]
        )  # (B, T, J)

        # Target encoder: full sequence, no gradient
        with torch.no_grad():
            target_all = self.target_encoder.encode_tokens(x)  # (B, T*J+1, E)
            target_all = target_all[:, 1:].view(B, T, J, -1)   # (B, T, J, E)

        # Context encoder: masked input (zero-out masked joints)
        x_masked = x.clone()
        x_masked[masks] = 0.0
        ctx_all = self.context_encoder.encode_tokens(x_masked)  # (B, T*J+1, E)
        ctx_all = ctx_all[:, 1:].view(B, T, J, -1)              # (B, T, J, E)

        jepa_losses = []
        for b in range(B):
            m = masks[b]  # (T, J)
            n_masked = m.sum().item()
            n_ctx = (~m).sum().item()
            if n_masked == 0 or n_ctx == 0:
                continue

            ctx = ctx_all[b][~m]      # (N_ctx, E)
            tgt = target_all[b][m]    # (N_mask, E)

            # Positional encodings for masked positions
            t_idx, j_idx = m.nonzero(as_tuple=True)
            pos_enc = self.context_encoder.token_embedding.position_encoding(
                t_idx, j_idx, device
            )  # (N_mask, E)

            pred = self.predictor(
                ctx.unsqueeze(0),      # (1, N_ctx, E)
                pos_enc.unsqueeze(0),  # (1, N_mask, E)
            )  # (1, N_mask, E)

            jepa_losses.append(F.mse_loss(pred.squeeze(0), tgt))

        jepa_loss = (
            torch.stack(jepa_losses).mean()
            if jepa_losses
            else x.new_zeros(1).squeeze()
        )

        # --- Motion-Gated TPC ---
        half = T // 2
        x_first = x[:, :half]   # (B, T//2, J, D)
        x_second = x[:, half:]  # (B, T//2, J, D)

        motion_mask = self._motion_gate(x_first)
        tpc_active_fraction = motion_mask.float().mean().item()

        tpc_loss = x.new_zeros(1).squeeze()
        if motion_mask.any():
            x_first_active = x_first[motion_mask]
            x_second_active = x_second[motion_mask]

            # Encode first half (variable T//2 — works because sinusoidal PE is length-agnostic)
            cls_half = self.context_encoder(x_first_active)       # (B_active, E)
            pred_second = self.tpc_decoder(cls_half)              # (B_active, T//2, J, D)
            tpc_loss = F.mse_loss(pred_second, x_second_active)

        total_loss = jepa_loss + 0.5 * tpc_loss

        return {
            "jepa_loss": jepa_loss,
            "tpc_loss": tpc_loss,
            "total_loss": total_loss,
            "tpc_active_fraction": tpc_active_fraction,
        }


class PoseViTClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible classifier wrapping PoseViT for score prediction.

    Can load pretrained weights from Pose-JEPA (kinescope pretrain).

    Parameters
    ----------
    embed_dim, n_layers, n_heads, seq_len, n_joints, coord_dim : int
        Architecture hyperparameters (must match pretrained checkpoint if loading one)
    lr : float
    epochs : int
    finetune_mode : str
        'linear_probe' — freeze encoder, train only MLP (best for N < 50)
        'full_finetune' — unfreeze last 2 layers + MLP (best for N >= 50)
        'scratch' — train from scratch, no encoder freezing
    pretrained_weights : str or None
        Path to checkpoint .pt from `kinescope pretrain`. If None, trains from scratch.
    batch_size : int
    device : str — 'auto', 'cpu', 'cuda', 'mps'
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 60,
        n_joints: int = N_JOINTS,
        coord_dim: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        finetune_mode: str = "linear_probe",
        pretrained_weights: Optional[str] = None,
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.n_joints = n_joints
        self.coord_dim = coord_dim
        self.lr = lr
        self.epochs = epochs
        self.finetune_mode = finetune_mode
        self.pretrained_weights = pretrained_weights
        self.batch_size = batch_size
        self.device = device

    def _get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build_model(self, n_classes: int) -> nn.ModuleDict:
        encoder = PoseViT(
            embed_dim=self.embed_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            seq_len=self.seq_len,
            n_joints=self.n_joints,
            coord_dim=self.coord_dim,
        )

        if self.pretrained_weights is not None:
            ckpt = torch.load(self.pretrained_weights, map_location="cpu")
            state = ckpt.get("context_encoder", ckpt)
            encoder.load_state_dict(state, strict=False)

        if self.finetune_mode == "linear_probe":
            for p in encoder.parameters():
                p.requires_grad_(False)
        elif self.finetune_mode == "full_finetune":
            for p in encoder.parameters():
                p.requires_grad_(False)
            n = len(encoder.transformer.layers)
            for layer in encoder.transformer.layers[max(0, n - 2) :]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            for p in encoder.norm.parameters():
                p.requires_grad_(True)
        # 'scratch': all parameters trainable (default)

        head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
        return nn.ModuleDict({"encoder": encoder, "head": head})

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape (N, T, J, D) — normalized pose sequences
        y : array-like, shape (N,) — class labels
        """
        device = self._get_device()

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        n_classes = len(self.label_encoder_.classes_)
        self.classes_ = self.label_encoder_.classes_

        self.model_ = self._build_model(n_classes).to(device)

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        params = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                enc_out = self.model_["encoder"](xb)
                logits = self.model_["head"](enc_out)
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        device = self._get_device()
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)

        self.model_.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i : i + self.batch_size].to(device)
                logits = self.model_["head"](self.model_["encoder"](xb))
                all_probs.append(F.softmax(logits, dim=-1).cpu())

        return torch.cat(all_probs).numpy()

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(proba.argmax(axis=1))
