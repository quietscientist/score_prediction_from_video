"""Intuitive plots for ep10 linear probe results."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path("results/probe_full_ep10/loso_metrics.json")
OUT = Path("results/probe_full_ep10")

m = json.load(open(RESULTS))
kin = m["kinematic_baseline"]
enc = m["combined"]
per_task = m["per_task_probes"]

TASKS = ["drinking", "communication", "la"]
TASK_LABELS = ["Drinking", "Communication", "Limb Agility"]
SUBJECTS = [str(f["subject"]) for f in kin["fold_results"]]

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
GRAY = "#8C8C8C"

# ── 1. Main summary: ρ and AUROC by task ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

metrics = [
    ("spearman", "Spearman ρ", [-0.1, 0.65]),
    ("auroc",    "AUROC",      [0.3, 0.95]),
]

for ax, (key, label, ylim) in zip(axes, metrics):
    kin_vals  = [kin["per_task"][t][key]  for t in TASKS]
    enc_vals  = [enc["per_task"][t][key]  for t in TASKS]
    # per-task probes
    pt_vals   = [per_task[t]["per_task"][t][key] for t in TASKS]

    x = np.arange(len(TASKS))
    w = 0.25
    ax.bar(x - w,   kin_vals, w, label="Kinematic baseline", color=GRAY,   alpha=0.85)
    ax.bar(x,       enc_vals, w, label="Encoder (combined)", color=BLUE,   alpha=0.85)
    ax.bar(x + w,   pt_vals,  w, label="Encoder (per-task)", color=ORANGE, alpha=0.85)

    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(TASK_LABELS, fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_ylim(ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

axes[0].set_title("Spearman ρ  (rank correlation with UPDRS score)", fontsize=11)
axes[1].set_title("AUROC  (high vs low severity discrimination)", fontsize=11)

fig.suptitle("Linear probe — ep10 checkpoint  |  LOSO cross-validation  (n=9 subjects)",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig1_task_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_task_summary.png")


# ── 2. Per-subject Spearman: kinematic vs encoder ────────────────────────────
kin_subj = {f["subject"]: f["spearman"] for f in kin["fold_results"]}
enc_subj = {f["subject"]: f["spearman"] for f in enc["fold_results"]}

subjs_sorted = sorted(SUBJECTS, key=lambda s: kin_subj[s])
kin_s = [kin_subj[s] for s in subjs_sorted]
enc_s = [enc_subj[s] for s in subjs_sorted]
delta = [e - k for e, k in zip(enc_s, kin_s)]
colors = [GREEN if d >= 0 else RED for d in delta]

fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(len(subjs_sorted))
ax.barh(y, kin_s, 0.38, label="Kinematic baseline", color=GRAY, alpha=0.85)
ax.barh(y + 0.38, enc_s, 0.38, label="Encoder (combined)", color=BLUE, alpha=0.85)
for i, (k, e, d) in enumerate(zip(kin_s, enc_s, delta)):
    sign = "+" if d >= 0 else ""
    ax.text(max(k, e) + 0.01, i + 0.19, f"{sign}{d:.2f}",
            va="center", fontsize=8, color=GREEN if d >= 0 else RED, fontweight="bold")

ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y + 0.19)
ax.set_yticklabels([f"Subject {s}" for s in subjs_sorted], fontsize=10)
ax.set_xlabel("Spearman ρ", fontsize=11)
ax.set_title("Per-subject Spearman ρ  —  Encoder vs Kinematic baseline\n"
             "(green Δ = encoder win, red Δ = encoder loss)", fontsize=11)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig2_per_subject.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_per_subject.png")


# ── 3. Loss curve ep1–10 ─────────────────────────────────────────────────────
lm = json.load(open("checkpoints/full_combined/metrics.json"))
epochs = list(range(1, len(lm["jepa_losses"]) + 1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, lm["jepa_losses"],       color=BLUE,   lw=2,   label="JEPA loss")
ax.plot(epochs, lm["invariant_losses"],  color=ORANGE, lw=2,   label="Invariant loss")
ax.plot(epochs, lm["total_losses"],      color="black", lw=2, ls="--", label="Total loss")
ax.plot(epochs, lm["sigreg_losses"],     color=GRAY,   lw=1.5, ls=":",  label="SIGReg loss")

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Pretraining loss curves  (1.44M clips, full dataset)", fontsize=11)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(epochs)
fig.tight_layout()
fig.savefig(OUT / "fig3_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_loss_curve.png")


# ── 4. Headline scorecard ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis("off")

rows = [
    ["",                    "Drinking", "Communication", "Limb Agility", "Mean"],
    ["Kinematic baseline",
        f"ρ={kin['per_task']['drinking']['spearman']:+.2f}",
        f"ρ={kin['per_task']['communication']['spearman']:+.2f}",
        f"ρ={kin['per_task']['la']['spearman']:+.2f}",
        f"ρ={kin['aggregate']['spearman']['mean']:+.2f}"],
    ["Encoder (combined)",
        f"ρ={enc['per_task']['drinking']['spearman']:+.2f}  AUC={enc['per_task']['drinking']['auroc']:.2f}",
        f"ρ={enc['per_task']['communication']['spearman']:+.2f}  AUC={enc['per_task']['communication']['auroc']:.2f}",
        f"ρ={enc['per_task']['la']['spearman']:+.2f}  AUC={enc['per_task']['la']['auroc']:.2f}",
        f"ρ={enc['aggregate']['spearman']['mean']:+.2f}  AUC={enc['aggregate']['auroc']['mean']:.2f}"],
    ["Encoder (comm. probe)",
        "—",
        f"ρ={per_task['communication']['per_task']['communication']['spearman']:+.2f}  AUC={per_task['communication']['per_task']['communication']['auroc']:.2f}",
        "—",
        "—"],
]

table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)

# highlight encoder rows
for j in range(5):
    table[1, j].set_facecolor("#F0F4FF")
    table[2, j].set_facecolor("#E8F4E8")

ax.set_title("Summary scorecard  —  ep10 checkpoint  (LOSO, n=9 subjects)",
             fontsize=11, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(OUT / "fig4_scorecard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_scorecard.png")

print("\nAll figures saved to", OUT)
