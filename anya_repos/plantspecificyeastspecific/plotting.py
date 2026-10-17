#plotting.py
# Figure helpers used by the annotatedfeaturesrun*.py scripts.
# Nothing in this file changes the model; it only draws and saves figures.
# All figures are saved as SVG (vector) with Arial text so they can be edited in Illustrator.

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


# ---------------------------------------------------------------------------
# General figure style
# ---------------------------------------------------------------------------

def apply_figure_style(fontsize=7):
    """Arial, small font, thin lines, and editable text in the SVG files."""
    plt.style.use("default")
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams["svg.fonttype"] = "none"  # save text as text, not paths
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["xtick.minor.width"] = 0.5
    mpl.rcParams["ytick.minor.width"] = 0.5


def apply_font_size(fig, fontsize=7):
    """SHAP hardcodes font sizes 13/11, so we override all text after plotting."""
    for ax in fig.get_axes():
        ax.title.set_fontsize(fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fontsize)
        for text in ax.texts:
            text.set_fontsize(fontsize)


def save_current_shap_figure(path, fontsize=7, marker_size=5):
    """Save the figure SHAP just drew, with vector (not rasterized) points."""
    fig = plt.gcf()
    apply_font_size(fig, fontsize=fontsize)
    for ax in fig.get_axes():
        for collection in ax.collections:
            collection.set_rasterized(False)
            if isinstance(collection, PathCollection):
                collection.set_sizes([marker_size])
        for image in list(ax.images):
            image.set_rasterized(False)
        for artist in ax.get_children():
            if hasattr(artist, "set_rasterized"):
                artist.set_rasterized(False)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# SHAP figures
# ---------------------------------------------------------------------------

def save_shap_beeswarm(shap_values, X, path, cmap, max_display, figsize, height_scale,
                       fontsize=7, marker_size=5):
    """One dot per tile and feature. x = SHAP value (right = pushes toward class 1),
    color = feature value (dark = low, bright = high)."""
    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        cmap=cmap,
        plot_size=(figsize[0], figsize[1] * height_scale),
        max_display=max_display,
        show=False,
    )
    save_current_shap_figure(path, fontsize=fontsize, marker_size=marker_size)


def save_shap_bar(shap_values, X, path, max_display, figsize, fontsize=7, marker_size=5):
    """Mean |SHAP value| per feature (overall feature importance)."""
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        plot_size=figsize,
        max_display=max_display,
        show=False,
    )
    save_current_shap_figure(path, fontsize=fontsize, marker_size=marker_size)


def _shap_groups_by_feature_value(shap_values, X, max_display):
    """For each top feature, split tiles at the feature's median value and
    collect the SHAP values of the low-value and high-value tiles."""
    shap_values = np.asarray(shap_values)
    feature_names = np.asarray(X.columns)
    n_show = min(max_display, shap_values.shape[1])
    order = np.argsort(np.abs(shap_values).mean(axis=0))[-n_show:]
    names = feature_names[order]
    low_groups = []
    high_groups = []
    for col_i in order:
        values = pd.to_numeric(X.iloc[:, int(col_i)], errors="coerce").to_numpy()
        s = shap_values[:, col_i]
        median = np.nanmedian(values)
        low = s[values <= median]
        high = s[values > median]
        if low.size == 0:
            low = s
        if high.size == 0:
            high = s
        low_groups.append(low)
        high_groups.append(high)
    return names, low_groups, high_groups


def save_class_mean_shap_bars(shap_values, X, path, max_display, figsize, cmap, fontsize=7):
    """Mean SHAP value of tiles with low vs high values of each feature."""
    names, low_groups, high_groups = _shap_groups_by_feature_value(shap_values, X, max_display)
    low_means = np.array([g.mean() for g in low_groups])
    high_means = np.array([g.mean() for g in high_groups])

    cmap_obj = mpl.colormaps.get_cmap(cmap)
    low_color = cmap_obj(0.15)
    high_color = cmap_obj(0.85)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, low_means, color=low_color, height=0.65, label="low feature value")
    ax.barh(y_pos, high_means, color=high_color, height=0.65, label="high feature value")
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("mean SHAP value")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, handlelength=1.0, borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_font_size(fig, fontsize=fontsize)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_class_mean_shap_points(shap_values, X, path, max_display, figsize, cmap, fontsize=7):
    """Same as the bars, but as points with SD error bars."""
    names, low_groups, high_groups = _shap_groups_by_feature_value(shap_values, X, max_display)

    def mean_and_sd(groups):
        means = np.array([g.mean() for g in groups])
        sds = np.array([g.std(ddof=1) if g.size > 1 else 0.0 for g in groups])
        return means, sds

    low_means, low_sds = mean_and_sd(low_groups)
    high_means, high_sds = mean_and_sd(high_groups)

    cmap_obj = mpl.colormaps.get_cmap(cmap)
    low_color = cmap_obj(0.15)
    high_color = cmap_obj(0.85)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    offset = 0.18
    style = dict(fmt="o", markersize=2.5, capsize=0, elinewidth=0.5,
                 markeredgewidth=0.4, linestyle="none")
    ax.errorbar(low_means, y_pos + offset, xerr=low_sds, color=low_color,
                label="low feature value", **style)
    ax.errorbar(high_means, y_pos - offset, xerr=high_sds, color=high_color,
                label="high feature value", **style)
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("mean SHAP value ± SD")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, handlelength=1.0, borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_font_size(fig, fontsize=fontsize)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")



def save_shap_class_boxes(
    shap_values, X, y_pred, path, class_names, max_display, figsize, cmap, fontsize=7,
    rank_start=0,
):
    """Box plots of SHAP values split by predicted class, colored by feature median.

    For each feature: upper box = tiles classified as class 0, lower = class 1.
    Color is the percentile rank of that class's median feature value among all
    explained tiles. Boxes can cross zero.

    Features are ranked by mean |SHAP| (1 = strongest). Shows ranks
    (rank_start + 1) through max_display. Example: rank_start=0, max_display=8
    → top 8; rank_start=8, max_display=25 → ranks 9–25.
    """
    shap_values = np.asarray(shap_values)
    X_arr = np.asarray(X, dtype=float)
    y_pred = np.asarray(y_pred)
    feature_names = np.asarray(X.columns)

    n_end = min(max_display, shap_values.shape[1])
    n_start = min(max(rank_start, 0), n_end)
    if n_start >= n_end:
        print(f"Skipping {path}: no features in rank slice ({n_start}, {n_end}]")
        return

    # top n_end features, ascending (weakest of this set first → bottom of plot)
    top_ascending = np.argsort(np.abs(shap_values).mean(axis=0))[-n_end:]
    # drop the strongest n_start from the top of the axis (= end of this array)
    # keep ranks (n_start+1)..n_end → exclude last n_start entries if we want rest...
    # top_ascending = [rank_n_end, ..., rank_2, rank_1]
    # top 8 (ranks 1..8): top_ascending[-8:] if n_end>=8 and n_start==0 with max_display=8
    # For rank_start=0, max_display=8: top_ascending has 8 elems, keep all.
    # For rank_start=8, max_display=25: top_ascending has 25 elems [r25..r1];
    #   ranks 9..25 = top_ascending[:-8] = [r25..r9]
    n_keep_strongest_out = n_start  # number of strongest ranks to exclude from this panel
    if n_keep_strongest_out > 0:
        order = top_ascending[:-n_keep_strongest_out]
    else:
        order = top_ascending

    cmap_obj = mpl.colormaps.get_cmap(cmap)
    offset = 0.18

    fig, ax = plt.subplots(figsize=figsize)
    box_style = dict(
        vert=False,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        showcaps=False,
        boxprops=dict(linewidth=0.5),
        whiskerprops=dict(linewidth=0.5),
        medianprops=dict(color="black", linewidth=0.5),
    )
    for y_pos, feature_idx in enumerate(order):
        shap_col = shap_values[:, feature_idx]
        value_col = X_arr[:, feature_idx]
        for pred_class, shift in ((0, offset), (1, -offset)):
            in_class = y_pred == pred_class
            if not in_class.any():
                continue
            median_value = np.nanmedian(value_col[in_class])
            scaled = float(np.nanmean(value_col <= median_value))
            boxes = ax.boxplot(
                shap_col[in_class],
                positions=[y_pos + shift],
                **box_style,
            )
            boxes["boxes"][0].set_facecolor(cmap_obj(scaled))

    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(feature_names[order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("SHAP value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot([], [], color="0.3", linewidth=4, label=class_names[0] + " (upper)")
    ax.plot([], [], color="0.3", linewidth=4, label=class_names[1] + " (lower)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        borderaxespad=0,
    )

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap_obj),
        ax=ax,
        fraction=0.05,
        pad=0.03,
        aspect=30,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature median (percentile)", labelpad=-4)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(width=0.5, length=2)

    apply_font_size(fig, fontsize=fontsize)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}  (ranks {n_start + 1}–{n_end}, {len(order)} features)")




# ---------------------------------------------------------------------------
# Performance figures: metrics barplot, truth table (confusion matrix), ROC curve
# ---------------------------------------------------------------------------

def save_metrics_barplot(y_true, y_pred, path, class_names, figsize, fontsize=7):
    """Accuracy, then precision / recall / F1 for each class."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    values = [accuracy, precision[0], recall[0], f1[0], precision[1], recall[1], f1[1]]
    save_metrics_barplot_from_values(values, path, class_names, figsize, fontsize=fontsize)


def save_metrics_barplot_from_values(values, path, class_names, figsize, fontsize=7):
    """Same barplot, from 7 numbers already computed (e.g. averaged across batches):
    accuracy, precision/recall/F1 of class 0, precision/recall/F1 of class 1."""
    labels = [
        "Accuracy",
        f"Precision\n{class_names[0]}", f"Recall\n{class_names[0]}", f"F1\n{class_names[0]}",
        f"Precision\n{class_names[1]}", f"Recall\n{class_names[1]}", f"F1\n{class_names[1]}",
    ]
    colors = ["#4C4C4C"] + ["#7B3294"] * 3 + ["#008837"] * 3

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
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{value:.2f}", ha="center", va="bottom", fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_truth_table(y_true, y_pred, path, class_names, figsize, fontsize=7):
    """2x2 confusion matrix: rows = true class, columns = predicted class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    save_truth_table_from_cm(cm, path, class_names, figsize, fontsize=fontsize)


def save_truth_table_from_cm(cm, path, class_names, figsize, fontsize=7):
    """Same truth table, from a 2x2 matrix already computed (e.g. the mean across
    batches). Counts that are not whole numbers are shown with one decimal."""
    cm = np.asarray(cm, dtype=float)

    def format_count(value):
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.1f}"

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

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            dark_cell = cm[i, j] > threshold
            ax.text(j, i, format_count(cm[i, j]), ha="center", va="center",
                    fontsize=fontsize, color="white" if dark_cell else "black")
            ax.text(j, i + 0.28, cell_labels[i][j], ha="center", va="center",
                    fontsize=max(fontsize - 1, 5), color="white" if dark_cell else "#666666")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_roc_curve(y_true, y_score, path, figsize, fontsize=7, curve_color="#008837"):
    """ROC curve of P(class 1); the legend shows the area under the curve (AUC)."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.6, linestyle="--")
    ax.plot(fpr, tpr, color=curve_color, linewidth=1.0, label=f"AUC = {roc_auc:.3f}")
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
    ax.legend(fontsize=fontsize, frameon=False, loc="lower right",
              handlelength=1.0, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"ROC AUC: {roc_auc:.3f}")
    return roc_auc


def save_mean_roc_curve(batch_curves, path, figsize, fontsize=7, curve_color="#008837"):
    """Mean ROC curve across batches, with a ±1 SD band.

    batch_curves is a list with one (y_true, y_score) pair per batch. Each batch curve is
    put on the same 101 false-positive-rate points so the curves can be averaged.
    The legend shows the mean AUC ± SD across batches.
    """
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []
    for y_true, y_score in batch_curves:
        fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_score), pos_label=1)
        aucs.append(auc(fpr, tpr))
        batch_tpr = np.interp(mean_fpr, fpr, tpr)
        batch_tpr[0] = 0.0
        tprs.append(batch_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    sd_tpr = np.std(tprs, axis=0)
    mean_auc = float(np.mean(aucs))
    sd_auc = float(np.std(aucs))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.6, linestyle="--")
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - sd_tpr, 0.0), np.minimum(mean_tpr + sd_tpr, 1.0),
                    color=curve_color, alpha=0.2, linewidth=0)
    ax.plot(mean_fpr, mean_tpr, color=curve_color, linewidth=1.0,
            label=f"AUC = {mean_auc:.3f} ± {sd_auc:.3f}")
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
    ax.legend(fontsize=fontsize, frameon=False, loc="lower right",
              handlelength=1.0, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Mean ROC AUC across {len(aucs)} batches: {mean_auc:.3f} ± {sd_auc:.3f}")
    return mean_auc, sd_auc
