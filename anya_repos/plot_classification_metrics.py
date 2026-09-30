"""Performance metric barplots, confusion-matrix truth tables, and ROC curves."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


def save_performance_figures(
    y_true,
    y_pred,
    output_dir,
    class_names,
    figsize=(1.8, 1.8),
    fontsize=7,
    cm=None,
    metric_values=None,
):
    """Save metrics barplot and truth-table SVGs to output_dir.

    Pass ``cm`` and/or ``metric_values`` to plot pre-aggregated results
    (e.g. mean confusion counts / metrics across batches) instead of
    computing them from a single ``y_true`` / ``y_pred`` pair.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "performance_metrics_bar.svg"
    truth_path = output_dir / "truth_table.svg"

    # Barplot needs more width for per-class metric labels.
    bar_figsize = (max(figsize[0] * 2.5, 3.5), figsize[1])
    if metric_values is None:
        _save_metrics_barplot(
            y_true, y_pred, metrics_path, class_names, bar_figsize, fontsize
        )
    else:
        _save_metrics_barplot_from_values(
            metric_values, metrics_path, class_names, bar_figsize, fontsize
        )

    if cm is None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _save_truth_table_from_cm(
        cm, truth_path, class_names, figsize, fontsize
    )
    return metrics_path, truth_path


def _metric_labels(class_names):
    return [
        "Accuracy",
        f"Precision\n{class_names[0]}",
        f"Recall\n{class_names[0]}",
        f"F1\n{class_names[0]}",
        f"Precision\n{class_names[1]}",
        f"Recall\n{class_names[1]}",
        f"F1\n{class_names[1]}",
    ]


def _save_metrics_barplot(y_true, y_pred, path, class_names, figsize, fontsize):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    values = [
        acc,
        precision[0],
        recall[0],
        f1[0],
        precision[1],
        recall[1],
        f1[1],
    ]
    _save_metrics_barplot_from_values(
        values, path, class_names, figsize, fontsize
    )


def _save_metrics_barplot_from_values(values, path, class_names, figsize, fontsize):
    labels = _metric_labels(class_names)
    colors = [
        "#4C4C4C",
        "#7B3294",
        "#7B3294",
        "#7B3294",
        "#008837",
        "#008837",
        "#008837",
    ]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.7, linewidth=0.5, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=fontsize, width=0.5, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _format_cm_value(value):
    """Format confusion counts; show one decimal when averaging leaves a fraction."""
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def _save_truth_table(y_true, y_pred, path, class_names, figsize, fontsize):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _save_truth_table_from_cm(cm, path, class_names, figsize, fontsize)


def _save_truth_table_from_cm(cm, path, class_names, figsize, fontsize):
    cm = np.asarray(cm, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("True", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize, width=0.5, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                _format_cm_value(cm[i, j]),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if cm[i, j] > thresh else "black",
            )

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i + 0.28,
                cell_labels[i][j],
                ha="center",
                va="center",
                fontsize=max(fontsize - 1, 5),
                color="white" if cm[i, j] > thresh else "#666666",
            )

    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_roc_curve(
    y_true=None,
    y_score=None,
    output_dir=None,
    figsize=(1.8, 1.8),
    fontsize=7,
    positive_label=1,
    curve_color="#008837",
    batch_curves=None,
    path=None,
):
    """Save an ROC curve SVG.

    Pass a single ``y_true`` / ``y_score`` pair, or ``batch_curves`` as a list of
    ``(y_true, y_score)`` pairs to plot the mean ROC across batches (with ±1 SD
    band when more than one batch is provided).
    """
    if path is None:
        if output_dir is None:
            raise ValueError("Pass output_dir or path")
        path = Path(output_dir) / "roc_curve.svg"
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if batch_curves is None:
        if y_true is None or y_score is None:
            raise ValueError("Pass y_true and y_score, or batch_curves")
        batch_curves = [(y_true, y_score)]

    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []
    for yt, ys in batch_curves:
        yt = np.asarray(yt)
        ys = np.asarray(ys)
        fpr, tpr, _ = roc_curve(yt, ys, pos_label=positive_label)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs)) if len(aucs) > 1 else 0.0
    std_tpr = np.std(tprs, axis=0) if len(tprs) > 1 else None

    prev_sans = mpl.rcParams["font.sans-serif"]
    prev_family = mpl.rcParams["font.family"]
    prev_size = mpl.rcParams["font.size"]
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["svg.fonttype"] = "none"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.6, linestyle="--")
    if std_tpr is not None:
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
        ax.fill_between(
            mean_fpr,
            tpr_lower,
            tpr_upper,
            color=curve_color,
            alpha=0.2,
            linewidth=0,
        )
    if len(aucs) > 1:
        label = f"AUC = {mean_auc:.3f} ± {std_auc:.3f}"
    else:
        label = f"AUC = {mean_auc:.3f}"
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=curve_color,
        linewidth=1.0,
        label=label,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", fontsize=fontsize)
    ax.set_ylabel("True positive rate", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize, width=0.5, length=2)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        fontsize=fontsize,
        frameon=False,
        loc="lower right",
        handlelength=1.0,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)

    mpl.rcParams["font.sans-serif"] = prev_sans
    mpl.rcParams["font.family"] = prev_family
    mpl.rcParams["font.size"] = prev_size
    print(f"Saved: {path}")
    if len(aucs) > 1:
        print(
            f"Mean ROC AUC across {len(aucs)} batches: "
            f"{mean_auc:.3f} ± {std_auc:.3f}"
        )
    else:
        print(f"ROC AUC: {mean_auc:.3f}")
    return path, mean_auc, std_auc
