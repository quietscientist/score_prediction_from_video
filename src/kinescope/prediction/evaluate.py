"""
Evaluation plotting utilities for score prediction.
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)


def plot_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(range(len(names)), vals[::-1], align="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
