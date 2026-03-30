"""ROC curves for ep10 linear probe — per task and per model."""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

PREDS = Path("results/probe_full_ep10/predictions.json")
OUT   = Path("results/probe_full_ep10")

p = json.load(open(PREDS))

TASKS       = ["drinking", "communication", "la"]
TASK_LABELS = {"drinking": "Drinking", "communication": "Communication", "la": "Limb Agility"}
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
GRAY   = "#8C8C8C"

def roc(y_raw, y_pred):
    binary = (np.array(y_raw) > 0).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return None, None, None
    fpr, tpr, _ = roc_curve(binary, y_pred)
    return fpr, tpr, auc(fpr, tpr)


# ── Fig 1: One ROC per task, kinematic vs encoder side by side ──────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

for ax, task in zip(axes, TASKS):
    # kinematic baseline
    kin = p["kinematic"]
    idx = [i for i, t in enumerate(kin["tasks"]) if t == task]
    fpr, tpr, a = roc([kin["y_raw"][i] for i in idx],
                      [kin["y_pred"][i] for i in idx])
    if fpr is not None:
        ax.plot(fpr, tpr, color=GRAY, lw=2, label=f"Kinematic  AUC={a:.2f}")

    # encoder combined
    enc = p["encoder_combined"]
    idx = [i for i, t in enumerate(enc["tasks"]) if t == task]
    fpr, tpr, a = roc([enc["y_raw"][i] for i in idx],
                      [enc["y_pred"][i] for i in idx])
    if fpr is not None:
        ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Encoder (combined)  AUC={a:.2f}")

    # per-task encoder
    key = f"encoder_{task}"
    if key in p:
        pt = p[key]
        idx = [i for i, t in enumerate(pt["tasks"]) if t == task]
        fpr, tpr, a = roc([pt["y_raw"][i] for i in idx],
                          [pt["y_pred"][i] for i in idx])
        if fpr is not None:
            ax.plot(fpr, tpr, color=ORANGE, lw=2, ls="--",
                    label=f"Encoder (per-task)  AUC={a:.2f}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(TASK_LABELS[task], fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("ROC curves — ep10 checkpoint  |  High vs Low severity  (pooled LOSO)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig5_roc_per_task.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_roc_per_task.png")


# ── Fig 2: All tasks overlaid for encoder combined ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
TASK_COLORS = {"drinking": "#4C72B0", "communication": "#DD8452", "la": "#55A868"}

for ax, (key, label) in zip(axes, [("kinematic", "Kinematic baseline"),
                                    ("encoder_combined", "Encoder (combined)")]):
    d = p[key]
    for task, color in TASK_COLORS.items():
        idx = [i for i, t in enumerate(d["tasks"]) if t == task]
        fpr, tpr, a = roc([d["y_raw"][i] for i in idx],
                          [d["y_pred"][i] for i in idx])
        if fpr is not None:
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f"{TASK_LABELS[task]}  AUC={a:.2f}")

    # overall
    fpr, tpr, a = roc(d["y_raw"], d["y_pred"])
    if fpr is not None:
        ax.plot(fpr, tpr, color="black", lw=2, ls="--", label=f"Overall  AUC={a:.2f}")

    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("ROC curves by task  —  Kinematic baseline vs Encoder (ep10)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig6_roc_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6_roc_comparison.png")

print("\nDone.")
