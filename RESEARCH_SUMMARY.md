# Kinescope: Pose-JEPA for Clinical Movement Analysis

## Goals

Predict clinical severity scores from human movement videos using self-supervised pretraining on large unlabeled pose datasets, then linear probing on small labeled clinical datasets. Two target tasks:

- **UDysRS**: dyskinesia severity in Parkinson's disease (adult, 775 clips, UPDRS-style ordinal scores per body segment)
- **GMA**: fidgety movement quality in infants (907 clips, 3-class: F+ normal / F+/- sporadic / F- absent)

The core hypothesis is that a model pretrained to predict masked pose tokens learns movement representations that generalize to clinical scoring — without requiring labeled data during pretraining.

---

## Architecture

**Input**: COCO-17 keypoints (T×17×2), normalized to shoulder-centered unit scale via `normalize_clip`. Face joints (eyes/ears) are dropped; shoulder center defines origin; torso-to-shoulder distance defines scale.

**Pose-JEPA** (`src/kinescope/prediction/_vit.py`):

- **`PoseViT`**: transformer encoder with per-joint token embedding (linear projection of (x,y) coordinates) + sinusoidal temporal positional encoding + learned joint positional encoding. CLS token aggregates the sequence-level representation.
- **`JEPAPredictor`**: 2-layer cross-attention decoder. Takes context encoder embeddings of unmasked tokens and positional queries for masked positions; predicts what the target encoder would produce for those positions.
- **EMA target encoder** (τ = 0.9 → 0.996, cosine warmup over training): produces stable prediction targets without collapse. Updated each optimizer step.
- **Spatiotemporal block masking** (~50% ratio, limb joints 2× weighted): the context encoder sees unmasked tokens; the predictor reconstructs masked target encoder embeddings.

**Auxiliary losses** (all optional, additive, zero-weighted by default):

| Loss | Flag | Purpose |
|---|---|---|
| TPC | `--tpc-weight` | Motion-gated temporal prediction: predict second half of clip from first half. Gated to clips with sufficient motion. |
| Invariant | `--invariant-weight` | Predict 7 kinematic invariants (symmetry, smoothness, coordination, entropy) from CLS. Provides clinically-grounded structure. |
| SIGReg | `--sigreg-weight` | Sketched isotropic Gaussian regularization on CLS embeddings. Constrains representation geometry toward N(0,I). |
| Long-horizon | `--long-horizon-weight` | Coarse segment-level future latent prediction. |

**Downstream evaluation**: frozen encoder → mean CLS over sliding windows → ridge regression (UDysRS) or logistic regression with class balancing (GMA).

---

## Rationale and Sources

### Why JEPA over masked autoencoders or contrastive learning?

Masked autoencoders (MAE; He et al., NeurIPS 2022) reconstruct raw input coordinates, biasing the model toward low-level details. For skeleton data, predicting joint coordinates is trivial — linear interpolation resolves most masked frames. JEPA (Assran et al., ICLR 2023; Bardes et al., ICLR 2024 V-JEPA) predicts in *latent space*, encouraging abstract temporal reasoning rather than low-level reconstruction.

Contrastive methods (SimCLR, MoCo) require defining augmentation invariances. For clinical movement, the correct invariances are unknown — you want the model to be sensitive to dyskinesia-related features while being invariant to recording artifacts. Defining augmentations that respect this is non-trivial. JEPA avoids this by learning from prediction rather than invariance.

### Why EMA target encoder over SIGReg alone?

SIGReg (Balestriero et al., LeJEPA, 2024) replaces the EMA target encoder with a geometric regularizer that constrains embeddings to an isotropic Gaussian distribution, simplifying training by removing EMA scheduling. However, SIGReg's collapse prevention has only been validated at scale on large image and video benchmarks. In our small-data clinical regime (~1.44M pretraining clips vs. billions of frames in image SSL), EMA provides a proven, well-understood collapse safeguard (validated in MoCo v3, Chen & He 2021; BYOL, Grill et al. 2020). SIGReg is therefore kept as an *additive* regularizer for embedding geometry rather than an EMA replacement.

### Why spatiotemporal block masking?

Random token masking can be solved by local interpolation — the model learns to average neighbors without understanding movement. Block masking (contiguous temporal windows + body regions) forces the model to predict entire body segments over time, requiring genuine motion modeling. V-JEPA (Bardes et al., 2024) demonstrated this for video; our adaptation extends the block structure to joint-space with limb weighting that upsamples informationally rich distal joints (wrists, ankles).

### Why kinematic invariants as auxiliary targets?

Clinical assessors score movement quality via features like bilateral symmetry, movement smoothness, and upper-lower limb coordination — not raw coordinates. Predicting these as auxiliary targets (without labels) provides weak structural supervision that aligns the CLS representation with clinically meaningful dimensions before any labeled fine-tuning. Related conceptually to "self-distillation with structured priors" (DINO, Caron et al. 2021) and motion auxiliary prediction in video SSL (Tong et al., VideoMAE 2022).

### Pretraining data

1.44M clips from four domains covering broad human motion variability:

| Dataset | Source | Scale | Motion type |
|---|---|---|---|
| AMASS | SMPL mocap (40+ sub-datasets) | ~800K clips | Diverse everyday and sports |
| NTU RGB+D 120 | KinectV2, 120 action classes | ~400K clips | Structured actions, 2 subjects |
| HUMOTO | Mixamo character animations (FBX/GLB) | ~150K clips | Synthetic, broad styles |
| Penn Action | Videoclips with 2D pose | ~90K clips | Sports, fine-grained motion |

---

## Testable Hypotheses

### H1: Pretraining improves clinical linear probe
Pretrained encoder AUROC > random-init encoder AUROC on both GMA and UDysRS. The gap measures the value of pretraining. Expected direction is clear; magnitude is unknown.

### H2: SIGReg improves probe quality
`sigreg_weight > 0` yields higher downstream AUROC because N(0,I) embedding geometry maximizes linear separability for downstream logistic/ridge regression. Testable by ablating `sigreg_weight` at a fixed checkpoint epoch.

### H3: Kinematic invariants improve clinical alignment
`invariant_weight > 0` improves correlation with clinical scores relative to JEPA-only pretraining, because the auxiliary targets explicitly encode clinically-relevant movement dimensions. Testable by comparing `fixed_gradclip` (inv=0.5) vs. a JEPA-only run on the same data.

### H4: Unsupervised surprise tracks severity
JEPA reconstruction error (harder to predict = more unusual movement) correlates positively with GMA severity (Spearman ρ > 0) without any labeled data. This tests whether the model's implicit prediction difficulty aligns with clinical abnormality.

### H5: Encoder sensitivity concentrates on distal joints
Per-joint occlusion analysis predicts that wrists and ankles drive the CLS embedding more than proximal joints — consistent with GMA clinical criteria, which define fidgety movements by distal, small-amplitude, variable-speed motions. Preliminary results (ep0010 checkpoint) already support this (right_wrist L2 shift = 8.73, left_ankle = 7.65 vs. shoulder = 3.2–4.1).

### H6: Cross-domain pretraining diversity outperforms volume
A model trained on all 4 datasets should outperform models trained on any single dataset at matched clip count, because cross-domain motion variability forces more general representations that transfer better to clinical data.

### H7: Structural prior reduces labeled data requirement
By combining the pretrained encoder with anatomical structure (joint connectivity, symmetry constraints), the model should achieve equivalent clinical probe performance with fewer labeled examples than a model without structural prior. Testable via a learning curve: AUROC vs. fraction of labeled clinical data used for probing.

---

## Current Status

| Item | Status |
|---|---|
| Pretraining (30 epochs, 1.44M clips, seq_len=30) | Running — epoch 1 complete (JEPA=0.158, t=3106s) |
| GMA probe (ep0010 checkpoint) | AUROC 0.660 encoder vs 0.734 kinematic baseline |
| UDysRS probe | Run on ep0010; see `results/linearprobe/` |
| JEPA surprise (norm mode) | ρ=+0.057, AUROC=0.556 — SIGReg weight too low |
| Per-joint occlusion sensitivity | Distal joints dominate; minimal normal/abnormal difference |
| Full JEPA surprise (jepa mode) | Pending checkpoint with predictor saved (ep0005+) |
