import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import shap

SCRIPT_DIR = Path(__file__).resolve().parent
TF_LIBRARY_SCRIPTS = SCRIPT_DIR.parent.parent / "scripts"
sys.path.insert(0, str(TF_LIBRARY_SCRIPTS))
from plot_classification_metrics import save_performance_figures

DEFAULT_INPUT = SCRIPT_DIR / "repressors_final__mutated.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "shap_figures"
CLASS_NAMES = ("non-repressor", "repressor")
LABEL_COL = "label"
ID_COL = "#Feature"
TF_ID_COL = "tf_id"
QUALITY_COL = "quality"
Z_COL = "z_log_enrichment_mean"
SEQUENCE_COL_IDX = 14
N_REPRESSORS = 2000
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_N_FEATURES = 100
SCORE_CSV_NAME = "sequence_model_scores.csv"
# Mutation-tracking metadata; not biochemical features for the model.
EXCLUDE_FEATURE_COLS = {
    "protein_seq_wt",
    "n_motif_core_positions_mutated",
    "motif_mutation_success",
    "had_canonical_motif",
}


def apply_figure_style(fontsize=7):
    plt.style.use("default")
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["xtick.minor.width"] = 0.5
    mpl.rcParams["ytick.minor.width"] = 0.5


def apply_font_size(fig, fontsize=7):
    """SHAP hardcodes fontsize=13/11; override all text after plotting."""
    for ax in fig.get_axes():
        ax.title.set_fontsize(fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fontsize)
        for text in ax.texts:
            text.set_fontsize(fontsize)


def plot_class_mean_shap_bars(
    shap_values,
    y,
    feature_names,
    class_names,
    max_display,
    figsize,
    cmap,
    fontsize,
    path,
):
    """Average SHAP values by class and plot as diverging bars.

    Class 0 means typically point left; class 1 means typically point right.
    Feature order matches the beeswarm (mean |SHAP|).
    """
    shap_values = np.asarray(shap_values)
    y = np.asarray(y)
    feature_names = np.asarray(feature_names)

    n_show = min(max_display, shap_values.shape[1])
    order = np.argsort(np.abs(shap_values).mean(axis=0))[-n_show:]
    names = feature_names[order]
    class0_means = shap_values[y == 0][:, order].mean(axis=0)
    class1_means = shap_values[y == 1][:, order].mean(axis=0)

    cmap_obj = mpl.colormaps.get_cmap(cmap)
    class0_color = cmap_obj(0.15)
    class1_color = cmap_obj(0.85)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, class0_means, color=class0_color, height=0.65, label=class_names[0])
    ax.barh(y_pos, class1_means, color=class1_color, height=0.65, label=class_names[1])
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("mean SHAP value")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        borderaxespad=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_font_size(fig, fontsize=fontsize)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_class_mean_shap_points(
    shap_values,
    y,
    feature_names,
    class_names,
    max_display,
    figsize,
    cmap,
    fontsize,
    path,
):
    """Plot class-wise mean SHAP values as markers with SD error bars."""
    shap_values = np.asarray(shap_values)
    y = np.asarray(y)
    feature_names = np.asarray(feature_names)

    n_show = min(max_display, shap_values.shape[1])
    order = np.argsort(np.abs(shap_values).mean(axis=0))[-n_show:]
    names = feature_names[order]
    class0 = shap_values[y == 0][:, order]
    class1 = shap_values[y == 1][:, order]

    cmap_obj = mpl.colormaps.get_cmap(cmap)
    class0_color = cmap_obj(0.15)
    class1_color = cmap_obj(0.85)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    offset = 0.18
    err_style = dict(
        fmt="o",
        markersize=2.5,
        capsize=0,
        elinewidth=0.5,
        markeredgewidth=0.4,
        linestyle="none",
    )
    ax.errorbar(
        class0.mean(axis=0),
        y_pos + offset,
        xerr=class0.std(axis=0, ddof=1),
        color=class0_color,
        label=class_names[0],
        **err_style,
    )
    ax.errorbar(
        class1.mean(axis=0),
        y_pos - offset,
        xerr=class1.std(axis=0, ddof=1),
        color=class1_color,
        label=class_names[1],
        **err_style,
    )
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("mean SHAP value ± SD")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        borderaxespad=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_font_size(fig, fontsize=fontsize)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_shap_class_boxes(
    shap_values,
    X,
    y_pred,
    feature_names,
    class_names,
    max_display,
    figsize,
    cmap,
    fontsize,
    path,
):
    """Box plots of SHAP values split by predicted class, colored by median feature value.

    For each feature there are two boxes: tiles the model classified as class 0 (drawn
    slightly above the row) and tiles classified as class 1 (slightly below). X is the
    SHAP value of that feature; box color is the percentile rank of that class's median
    feature value among all explained tiles (fraction of tiles with a lower or equal
    value). Boxes can cross zero: a tile called "repressor" can still have a negative
    SHAP for one feature.
    """
    shap_values = np.asarray(shap_values)
    X = np.asarray(X, dtype=float)
    y_pred = np.asarray(y_pred)
    feature_names = np.asarray(feature_names)

    n_show = min(max_display, shap_values.shape[1])
    order = np.argsort(np.abs(shap_values).mean(axis=0))[-n_show:]
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
        value_col = X[:, feature_idx]
        for pred_class, shift in ((0, offset), (1, -offset)):
            in_class = y_pred == pred_class
            if not in_class.any():
                continue
            median_value = np.nanmedian(value_col[in_class])
            # Percentile rank of the class median among all tiles (0 = lowest, 1 = highest).
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

    # Tiny legend: which row offset is which predicted class
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


def save_vector_svg(path, fontsize=7, marker_size=5):
    """Force vector markers and override SHAP's hardcoded font/marker sizes."""
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


def class1_shap(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values[:, :, 1]


def select_global_features(repressors, nonrepressors, feature_cols, n_features):
    """Pick a fixed top-N feature set once so SHAP columns align across batches."""
    seed = RANDOM_STATE
    n_rep = min(N_REPRESSORS, len(repressors))
    n_non = min(n_rep, len(nonrepressors))
    sample_df = pd.concat(
        [
            repressors.sample(n=n_rep, replace=False, random_state=seed),
            nonrepressors.sample(n=n_non, replace=False, random_state=seed),
        ],
        ignore_index=True,
    )
    X = sample_df[feature_cols]
    y = sample_df[LABEL_COL]
    nan_cols = X.columns[X.isna().any()].tolist()
    X = X.drop(columns=nan_cols)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X, y)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    top_features = importance_df.head(n_features)["Feature"].tolist()
    return top_features, nan_cols, importance_df


def train_eval_shap_batch(batch_df, full_df, top_features, seed, batch_number):
    X = batch_df[top_features]
    y = batch_df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    acc = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, labels=[0, 1], zero_division=0
    )
    metrics = np.array([
        acc,
        precision[0],
        recall[0],
        f1[0],
        precision[1],
        recall[1],
        f1[1],
    ], dtype=float)

    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=seed)
    y_sample = y_test.loc[X_sample.index]
    pred_sample = pd.Series(
        model.predict(X_sample),
        index=X_sample.index,
    )
    shap_for_plot = class1_shap(explainer.shap_values(X_sample))
    mean_abs_shap = pd.Series(
        np.abs(shap_for_plot).mean(axis=0),
        index=X_sample.columns,
    )

    # Score every sequence with this batch model so IDs can be averaged later.
    X_all = full_df[top_features].copy()
    train_medians = X_train.median()
    X_all = X_all.fillna(train_medians).fillna(0)
    proba = model.predict_proba(X_all)
    class1_idx = list(model.classes_).index(1)
    score_df = pd.DataFrame({
        ID_COL: full_df[ID_COL].values,
        TF_ID_COL: full_df[TF_ID_COL].values if TF_ID_COL in full_df.columns else np.nan,
        QUALITY_COL: full_df[QUALITY_COL].values if QUALITY_COL in full_df.columns else np.nan,
        Z_COL: full_df[Z_COL].values if Z_COL in full_df.columns else np.nan,
        LABEL_COL: full_df[LABEL_COL].values,
        "batch": batch_number,
        "score": proba[:, class1_idx],
    })

    return {
        "cm": cm,
        "metrics": metrics,
        "predictions": predictions,
        "y_test": y_test,
        "X_sample": X_sample.reset_index(drop=True),
        "y_sample": y_sample.reset_index(drop=True),
        "pred_sample": pred_sample.reset_index(drop=True),
        "shap_values": shap_for_plot,
        "mean_abs_shap": mean_abs_shap,
        "score_df": score_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train repressor vs non-repressor models across non-repressor "
            "batches and save SHAP / performance plots."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to combined repressor dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save SHAP figures.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1.8, 1.8),
        help="Figure size in inches (width height). Default: 1.8 1.8",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap for the beeswarm plot. Default: coolwarm",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=15,
        help="Number of top features to show in SHAP plots. Default: 15",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=7,
        help="Font size for all plot text (pt). Default: 7",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=5,
        help="Beeswarm marker area in points^2 (matplotlib scatter s). Default: 5",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help=(
            "Number of non-repressor batches. Default: "
            "floor(n_nonrepressors / n_repressors)."
        ),
    )
    parser.add_argument(
        "--beeswarm-sample",
        type=int,
        default=1000,
        help=(
            "Max points in the beeswarm SVG. Pooled SHAP is downsampled "
            "to this many rows so Illustrator can open the file. Default: 1000"
        ),
    )
    return parser.parse_args()


args = parse_args()
input_path = args.input.expanduser().resolve()
output_dir = args.output_dir.expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
figsize = tuple(args.figsize)
cmap = args.cmap
max_display = args.max_display
font_size = args.font_size
marker_size = args.marker_size
beeswarm_sample = args.beeswarm_sample
apply_figure_style(fontsize=font_size)

if not input_path.is_file():
    raise FileNotFoundError(
        f"Input dataset not found: {input_path}\n"
        "Run filterrepressors.py first, or pass the CSV path with --input."
    )

dataset = pd.read_csv(input_path)
print("Dataset shape:", dataset.shape)
print("\nLabel distribution:")
print(dataset[LABEL_COL].value_counts().sort_index())

repressors = dataset[dataset[LABEL_COL] == 1].copy()
nonrepressors = dataset[dataset[LABEL_COL] == 0].copy()

n_repressors_to_use = min(N_REPRESSORS, len(repressors))
n_nonrepressors_to_use = min(n_repressors_to_use, len(nonrepressors))
n_batches = args.n_batches
if n_batches is None:
    n_batches = max(1, len(nonrepressors) // max(len(repressors), 1))

candidate_features = list(dataset.columns[SEQUENCE_COL_IDX + 1:-1])
excluded = [col for col in candidate_features if col in EXCLUDE_FEATURE_COLS]
if excluded:
    print("Dropped mutation-metadata columns:", excluded)
    candidate_features = [
        col for col in candidate_features if col not in EXCLUDE_FEATURE_COLS
    ]
non_numeric = [
    col for col in candidate_features
    if not pd.api.types.is_numeric_dtype(dataset[col])
]
if non_numeric:
    print("Dropped non-numeric feature columns:", non_numeric)
    candidate_features = [
        col for col in candidate_features if col not in non_numeric
    ]

print("\nSelecting global top features for aligned SHAP across batches...")
top_features, dropped_nan_cols, importance_df = select_global_features(
    repressors, nonrepressors, candidate_features, TOP_N_FEATURES
)
print("Dropped NaN columns:", len(dropped_nan_cols))
print("Selected features:", len(top_features))
print("\nTop 10 RF features:")
print(importance_df.head(10))

print("\nBATCH SETTINGS")
print("Number of batches:", n_batches)
print("Repressors per batch:", n_repressors_to_use)
print("Non-repressors per batch:", n_nonrepressors_to_use)
print("Sequence column:", dataset.columns[SEQUENCE_COL_IDX])

batch_cms = []
batch_metrics = []
batch_mean_abs_shap = []
pooled_shap = []
pooled_X = []
pooled_y = []
pooled_pred = []
batch_scores = []
last_result = None

for batch_number in range(1, n_batches + 1):
    print("\n" + "=" * 60)
    print(f"BATCH {batch_number:02d}")
    print("=" * 60)

    seed = RANDOM_STATE + batch_number
    repressors_batch = repressors.sample(
        n=n_repressors_to_use,
        replace=False,
        random_state=seed,
    )
    nonrepressors_batch = nonrepressors.sample(
        n=n_nonrepressors_to_use,
        replace=False,
        random_state=seed,
    )
    batch_df = pd.concat(
        [repressors_batch, nonrepressors_batch],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(
        f"{len(repressors_batch)} repressors + "
        f"{len(nonrepressors_batch)} non-repressors = {len(batch_df)} total"
    )

    result = train_eval_shap_batch(
        batch_df, dataset, top_features, seed, batch_number
    )
    batch_cms.append(result["cm"])
    batch_metrics.append(result["metrics"])
    batch_mean_abs_shap.append(result["mean_abs_shap"])
    pooled_shap.append(result["shap_values"])
    pooled_X.append(result["X_sample"])
    pooled_y.append(result["y_sample"])
    pooled_pred.append(result["pred_sample"])
    batch_scores.append(result["score_df"])
    last_result = result

    print("Confusion matrix:\n", result["cm"])
    print("Accuracy:", round(result["metrics"][0], 5))

mean_cm = np.mean(np.stack(batch_cms, axis=0), axis=0)
mean_metrics = np.mean(np.stack(batch_metrics, axis=0), axis=0)
mean_abs_shap = (
    pd.concat(batch_mean_abs_shap, axis=1)
    .mean(axis=1)
    .sort_values(ascending=False)
)

print("\nMean confusion matrix across batches:")
print(mean_cm)
print("\nMean metrics across batches:")
print(
    "Accuracy={:.3f}, "
    "Prec0={:.3f}, Rec0={:.3f}, F1_0={:.3f}, "
    "Prec1={:.3f}, Rec1={:.3f}, F1_1={:.3f}".format(*mean_metrics)
)
print("\nTop 15 mean |SHAP| features across batches:")
print(mean_abs_shap.head(15))

print("\nSaving sequence scores (mean P(repressor) across batches)...")
scores_path = output_dir / SCORE_CSV_NAME
all_scores = pd.concat(batch_scores, ignore_index=True)
sequence_scores = (
    all_scores.groupby(ID_COL, as_index=False)
    .agg(
        tf_id=(TF_ID_COL, "first"),
        quality=(QUALITY_COL, "first"),
        z_log_enrichment_mean=(Z_COL, "first"),
        label=(LABEL_COL, "first"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        n_batches=("batch", "nunique"),
    )
    .sort_values(["label", "mean_score"], ascending=[False, False])
    .reset_index(drop=True)
)
sequence_scores.to_csv(scores_path, index=False)
print(f"Saved: {scores_path}")
print("Sequences scored:", len(sequence_scores))
print(sequence_scores.head(10).to_string(index=False))

print("\nSaving performance figures (batch-averaged)...")
save_performance_figures(
    last_result["y_test"],
    last_result["predictions"],
    output_dir,
    class_names=CLASS_NAMES,
    figsize=figsize,
    fontsize=font_size,
    cm=mean_cm,
    metric_values=mean_metrics,
)

# Pool SHAP values from all batches; order features by batch-averaged mean |SHAP|.
feature_order = mean_abs_shap.index.tolist()
shap_for_plot = np.vstack(pooled_shap)
X_for_plot = pd.concat(pooled_X, ignore_index=True)[feature_order]
y_for_plot = pd.concat(pooled_y, ignore_index=True)
pred_for_plot = pd.concat(pooled_pred, ignore_index=True)
# Reorder SHAP columns to match averaged importance ranking.
col_index = {feat: i for i, feat in enumerate(top_features)}
order_idx = [col_index[feat] for feat in feature_order]
shap_for_plot = shap_for_plot[:, order_idx]

print(
    f"\nRunning aggregated SHAP "
    f"({shap_for_plot.shape[0]} samples from {n_batches} batches)..."
)

n_beeswarm = min(beeswarm_sample, shap_for_plot.shape[0])
if n_beeswarm < shap_for_plot.shape[0]:
    rng = np.random.default_rng(RANDOM_STATE)
    beeswarm_idx = np.sort(
        rng.choice(shap_for_plot.shape[0], size=n_beeswarm, replace=False)
    )
    shap_beeswarm = shap_for_plot[beeswarm_idx]
    X_beeswarm = X_for_plot.iloc[beeswarm_idx]
    print(
        f"Downsampled beeswarm from {shap_for_plot.shape[0]} "
        f"to {n_beeswarm} points"
    )
else:
    shap_beeswarm = shap_for_plot
    X_beeswarm = X_for_plot

summary_path = output_dir / "shap_summary_beeswarm.svg"
bar_path = output_dir / "shap_summary_bar.svg"
class_mean_path = output_dir / "shap_summary_class_mean_bars.svg"
class_mean_points_path = output_dir / "shap_summary_class_mean_points.svg"
class_boxes_path = output_dir / "shap_summary_class_boxes.svg"

shap.summary_plot(
    shap_beeswarm,
    X_beeswarm,
    plot_type="dot",
    cmap=cmap,
    plot_size=figsize,
    max_display=max_display,
    show=False,
)
save_vector_svg(summary_path, fontsize=font_size, marker_size=marker_size)
print(f"Saved: {summary_path}")

shap.summary_plot(
    shap_for_plot,
    X_for_plot,
    plot_type="bar",
    plot_size=figsize,
    max_display=max_display,
    show=False,
)
save_vector_svg(bar_path, fontsize=font_size, marker_size=marker_size)
print(f"Saved: {bar_path}")

plot_class_mean_shap_bars(
    shap_for_plot,
    y_for_plot,
    X_for_plot.columns,
    CLASS_NAMES,
    max_display=max_display,
    figsize=figsize,
    cmap=cmap,
    fontsize=font_size,
    path=class_mean_path,
)
print(f"Saved: {class_mean_path}")

plot_class_mean_shap_points(
    shap_for_plot,
    y_for_plot,
    X_for_plot.columns,
    CLASS_NAMES,
    max_display=max_display,
    figsize=(figsize[0] * 0.85 * 0.85, figsize[1]),
    cmap=cmap,
    fontsize=font_size,
    path=class_mean_points_path,
)
print(f"Saved: {class_mean_points_path}")

plot_shap_class_boxes(
    shap_for_plot,
    X_for_plot,
    pred_for_plot,
    X_for_plot.columns,
    CLASS_NAMES,
    max_display=15,
    figsize=figsize,
    cmap="PRGn",
    fontsize=font_size,
    path=class_boxes_path,
)
print(f"Saved: {class_boxes_path}")

print(f"\nFigures written to: {output_dir}")
print(
    f"n_batches={n_batches}, figsize={figsize}, cmap={cmap}, "
    f"max_display={max_display}, font_size={font_size}, "
    f"marker_size={marker_size}, beeswarm_sample={beeswarm_sample}"
)
print(
    "Truth table / metrics / SHAP: aggregated across batches "
    "(fixed top features; SHAP pooled over all batch test samples)."
)
print(
    f"Sequence scores: mean P(repressor) from all {n_batches} batch models "
    f"written to {SCORE_CSV_NAME}."
)
