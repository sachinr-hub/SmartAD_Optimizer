import os
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)


ArrayLike = Union[np.ndarray, List[int], List[float]]


def ensure_results_dir(path: str = "results") -> str:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    labels: Optional[Iterable] = None,
) -> Dict[str, float]:
    """
    Compute common classification metrics.

    Parameters
    - y_true: ground-truth labels
    - y_pred: predicted labels (discrete classes)
    - average: 'binary', 'micro', 'macro', or 'weighted'
    - labels: optional label ordering for multi-class

    Returns
    - dict with accuracy, precision, recall, f1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # Detailed per-class report when appropriate
    try:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=labels)
        metrics["classification_report"] = report_dict  # type: ignore
    except Exception:
        # Some settings may fail (e.g., labels mismatch); ignore detailed report
        pass

    return metrics


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[List[Union[int, str]]] = None,
    display_labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    cmap: str = "Blues",
    out_dir: str = "results",
    file_prefix: str = "confusion_matrix",
) -> str:
    """
    Plot and save a confusion matrix.

    Returns path to the saved image.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine computation labels and display labels robustly
    y_true_vals = np.unique(y_true)
    y_pred_vals = np.unique(y_pred)
    inferred_classes = sorted(np.unique(np.concatenate([y_true_vals, y_pred_vals])).tolist())

    compute_labels: Optional[List[Union[int, str]]] = None
    tick_labels: List[str]

    if labels is not None:
        # If provided labels intersect with y_true values, use them directly for computation
        if any(l in set(y_true_vals.tolist()) for l in labels):
            compute_labels = labels
            tick_labels = [str(l) for l in labels]
        else:
            # Labels appear to be just display names; compute without specifying labels
            compute_labels = None
            tick_labels = [str(l) for l in labels]
    else:
        compute_labels = inferred_classes
        tick_labels = [str(l) for l in inferred_classes]

    cm = confusion_matrix(y_true, y_pred, labels=compute_labels)
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    if display_labels is not None and len(display_labels) == cm.shape[0]:
        tick_labels = [str(l) for l in display_labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(tick_labels)),
        yticks=np.arange(len(tick_labels)),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    if title is None:
        title = "Confusion Matrix" + (" (Normalized)" if normalize else "")
    ax.set_title(title)
    fig.tight_layout()

    ensure_results_dir(out_dir)
    ts = _timestamp()
    path = os.path.join(out_dir, f"{file_prefix}_{'norm_' if normalize else ''}{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


essential_curve_colors = {
    "roc_curve": "tab:blue",
    "pr_curve": "tab:green",
}


def plot_roc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: Optional[Union[int, str]] = None,
    out_dir: str = "results",
    file_prefix: str = "roc_curve",
    title: str = "ROC Curve",
) -> str:
    """
    Plot ROC curve for binary classification.

    y_score should be probability estimates or decision function for the positive class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=essential_curve_colors["roc_curve"], lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="tab:red", lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()

    ensure_results_dir(out_dir)
    ts = _timestamp()
    path = os.path.join(out_dir, f"{file_prefix}_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_precision_recall(
    y_true: ArrayLike,
    y_score: ArrayLike,
    out_dir: str = "results",
    file_prefix: str = "precision_recall_curve",
    title: str = "Precision-Recall Curve",
) -> str:
    """
    Plot Precision-Recall curve for binary classification.

    y_score should be probability estimates or decision function for the positive class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=essential_curve_colors["pr_curve"], lw=2, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()

    ensure_results_dir(out_dir)
    ts = _timestamp()
    path = os.path.join(out_dir, f"{file_prefix}_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metrics_bar(
    metrics: Dict[str, float],
    out_dir: str = "results",
    file_prefix: str = "metrics_bar",
    title: str = "Classification Metrics",
) -> str:
    """
    Plot a simple bar chart for accuracy, precision, recall, and f1.
    Ignores nested keys like 'classification_report'.
    """
    keys = [k for k in ["accuracy", "precision", "recall", "f1"] if k in metrics]
    vals = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=3)
    fig.tight_layout()

    ensure_results_dir(out_dir)
    ts = _timestamp()
    path = os.path.join(out_dir, f"{file_prefix}_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_classification_report_plots(
    y_true: ArrayLike,
    y_pred_labels: ArrayLike,
    y_scores: Optional[ArrayLike] = None,
    labels: Optional[List[str]] = None,
    average: str = "binary",
    out_dir: str = "results",
    prefix: str = "classification",
) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    High-level helper to compute metrics and generate common plots.

    Returns a dictionary with metric values and file paths of saved plots.
    """
    ensure_results_dir(out_dir)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred_labels,
        average=average,
        labels=labels,
    )

    ts = _timestamp()
    outputs: Dict[str, Union[str, Dict[str, float]]] = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    # Confusion matrices
    outputs["confusion_matrix_path"] = plot_confusion_matrix(
        y_true, y_pred_labels, labels=labels, normalize=False,
        out_dir=out_dir, file_prefix=f"{prefix}_cm"
    )
    outputs["confusion_matrix_normalized_path"] = plot_confusion_matrix(
        y_true, y_pred_labels, labels=labels, normalize=True,
        out_dir=out_dir, file_prefix=f"{prefix}_cm"
    )

    # Metrics bar
    outputs["metrics_bar_path"] = plot_metrics_bar(
        metrics, out_dir=out_dir, file_prefix=f"{prefix}_metrics"
    )

    # Curves (if scores provided)
    if y_scores is not None:
        try:
            outputs["roc_curve_path"] = plot_roc(
                y_true, y_scores, out_dir=out_dir, file_prefix=f"{prefix}_roc"
            )
        except Exception:
            pass
        try:
            outputs["pr_curve_path"] = plot_precision_recall(
                y_true, y_scores, out_dir=out_dir, file_prefix=f"{prefix}_pr"
            )
        except Exception:
            pass

    return outputs


if __name__ == "__main__":
    # Demo with synthetic binary classification labels
    rng = np.random.RandomState(42)
    n = 300
    y_true_demo = rng.randint(0, 2, size=n)
    # Pretend these are predicted probabilities
    y_scores_demo = rng.rand(n) * 0.7 + 0.15  # centered in [0.15, 0.85]
    threshold = 0.5
    y_pred_demo = (y_scores_demo >= threshold).astype(int)

    out = generate_classification_report_plots(
        y_true_demo,
        y_pred_demo,
        y_scores_demo,
        labels=["neg", "pos"],  # just for axis labels; optional
        average="binary",
        out_dir="results",
        prefix="demo",
    )
    print("Saved artifacts:\n", {k: v for k, v in out.items() if isinstance(v, str)})
