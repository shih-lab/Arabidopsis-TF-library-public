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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap

#annotatedfeaturesrunyeast.py

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from plot_classification_metrics import save_performance_figures, save_roc_curve

DEFAULT_INPUT = SCRIPT_DIR / "June26yeast_combined_activator_dataset1_finalData1.csv"
SHARED_FEATURES_FILE = SCRIPT_DIR.parent.parent / "data" / "plant_TF_all_data_annotated_features.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "shap_figures"
CLASS_NAMES = ("yeast-only", "shared")
SWARM_MAX_DISPLAY = 25
SWARM_CMAP = "magma"
SWARM_HEIGHT_SCALE = 1.75
REDUNDANT_DISORDER_COL = "Mean Disorder"


def apply_figure_style(fontsize=7):
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


def _shap_groups_by_feature_value(shap_values, X, feature_names, max_display):
    """Split each feature at its median into low vs high value groups."""
    shap_values = np.asarray(shap_values)
    X = pd.DataFrame(X)
    feature_names = np.asarray(feature_names)
    n_show = min(max_display, shap_values.shape[1])
    order = np.argsort(np.abs(shap_values).mean(axis=0))[-n_show:]
    names = feature_names[order]
    low_groups = []
    high_groups = []
    for col_i in order:
        name = feature_names[col_i]
        if name in X.columns:
            vals = pd.to_numeric(X[name], errors="coerce").to_numpy()
        else:
            vals = pd.to_numeric(X.iloc[:, int(col_i)], errors="coerce").to_numpy()
        s = shap_values[:, col_i]
        med = np.nanmedian(vals)
        low = s[vals <= med]
        high = s[vals > med]
        if low.size == 0:
            low = s
        if high.size == 0:
            high = s
        low_groups.append(low)
        high_groups.append(high)
    return names, low_groups, high_groups


def plot_class_mean_shap_bars(
    shap_values,
    X,
    feature_names,
    max_display,
    figsize,
    cmap,
    fontsize,
    path,
):
    """Average SHAP values for low vs high feature values.

    Each feature is split at its median. Coolwarm blue = low feature value,
    red = high, matching the beeswarm colormap.
    """
    names, low_groups, high_groups = _shap_groups_by_feature_value(
        shap_values, X, feature_names, max_display
    )
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
    X,
    feature_names,
    max_display,
    figsize,
    cmap,
    fontsize,
    path,
):
    """Mean SHAP for low vs high feature values, with SD error bars."""
    names, low_groups, high_groups = _shap_groups_by_feature_value(
        shap_values, X, feature_names, max_display
    )

    def _mean_std(groups):
        means = np.array([g.mean() for g in groups])
        stds = np.array([g.std(ddof=1) if g.size > 1 else 0.0 for g in groups])
        return means, stds

    low_means, low_stds = _mean_std(low_groups)
    high_means, high_stds = _mean_std(high_groups)

    cmap_obj = mpl.colormaps.get_cmap(cmap)
    low_color = cmap_obj(0.15)
    high_color = cmap_obj(0.85)

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
        low_means,
        y_pos + offset,
        xerr=low_stds,
        color=low_color,
        label="low feature value",
        **err_style,
    )
    ax.errorbar(
        high_means,
        y_pos - offset,
        xerr=high_stds,
        color=high_color,
        label="high feature value",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train yeast-specific vs shared activator model and save SHAP plots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to combined activator dataset CSV.",
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
apply_figure_style(fontsize=font_size)

if not input_path.is_file():
    raise FileNotFoundError(
        f"Input dataset not found: {input_path}\n"
        "Pass the CSV path with --input /path/to/"
        "June26yeast_combined_activator_dataset1_finalData1.csv"
    )

# Load dataset
dataset = pd.read_csv(input_path)
if REDUNDANT_DISORDER_COL in dataset.columns:
    dataset = dataset.drop(columns=[REDUNDANT_DISORDER_COL])
    print(f"Dropped redundant column: {REDUNDANT_DISORDER_COL}")
if SHARED_FEATURES_FILE.is_file() and "#Feature" in dataset.columns:
    latest = pd.read_csv(
        SHARED_FEATURES_FILE,
        usecols=["#Feature", "tile_context_disorder", "tile_intrinsic_disorder"],
    ).drop_duplicates(subset=["#Feature"], keep="first")
    latest_idx = latest.set_index("#Feature")
    for col in ("tile_context_disorder", "tile_intrinsic_disorder"):
        mapped = dataset["#Feature"].map(latest_idx[col])
        if col in dataset.columns:
            dataset[col] = mapped.fillna(dataset[col])
        else:
            dataset[col] = mapped
    print(f"Disorder columns taken from {SHARED_FEATURES_FILE}")

print("Dataset shape:", dataset.shape)

label_col = dataset.columns[-1]

n_class0 = (dataset[label_col] == 0).sum()
class0 = dataset[dataset[label_col] == 0].sample(
    n=min(800, n_class0),
    random_state=42
)

class1 = dataset[dataset[label_col] == 1]

dataset = pd.concat([class0, class1], axis=0)

# Shuffle
dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nDataset after class-0 subsampling:")
print(dataset[label_col].value_counts().sort_index())
print("New dataset shape:", dataset.shape)

# Column P = index 15 (0-based), hence I put index 14
sequence_col_idx = 14

# Last column is label
label_col = dataset.columns[-1]

# Features are everything after and including column P up to but not including label(the last column)
feature_cols = dataset.columns[sequence_col_idx + 1:-1]

print("\nSequence column:", dataset.columns[sequence_col_idx])
print("Label column:", label_col)
print("Number of features:", len(feature_cols))

X = dataset[feature_cols]
y = dataset[label_col]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Find columns with NaNs using training data only
nan_counts = X_train.isna().sum()
nan_cols = nan_counts[nan_counts > 0]

# Drop those columns from both train and test
X_train = X_train.drop(columns=nan_cols.index)
X_test = X_test.drop(columns=nan_cols.index)

print("Dropped", len(nan_cols), "columns.")
print("Remaining features:", X_train.shape[1])
print("\nDropped columns:")
for col in nan_cols.index:
    print(col)

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())



model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)


# Select Top 300 features using RF importance

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

top300_features = importance_df.head(100)["Feature"].tolist()

print("\nTop 10 RF Features:")
print(importance_df.head(10))

print("\nNumber of selected features:", len(top300_features))



#predictions = model.predict(X_test)
# Keep only top 300 features

X_train_top300 = X_train[top300_features]
X_test_top300 = X_test[top300_features]

# Train new RF

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_top300, y_train)

predictions = model.predict(X_test_top300)
class1_idx = list(model.classes_).index(1)
y_score = model.predict_proba(X_test_top300)[:, class1_idx]



print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nSaving performance figures...")
metrics_path, truth_path = save_performance_figures(
    y_test,
    predictions,
    output_dir,
    class_names=CLASS_NAMES,
    figsize=figsize,
    fontsize=font_size,
)
save_roc_curve(
    y_true=y_test,
    y_score=y_score,
    output_dir=output_dir,
    figsize=figsize,
    fontsize=font_size,
)

# SHAP
print("\nRunning SHAP...")

explainer = shap.TreeExplainer(model)

#X_sample = X_test.sample(
 #   n=min(1000, len(X_test)),
  #  random_state=42
#)

X_sample = X_test_top300.sample(
    n=min(1000, len(X_test_top300)),
    random_state=42
)
y_sample = y_test.loc[X_sample.index]
shap_values = explainer.shap_values(X_sample)

# Binary classification
if isinstance(shap_values, list):
    shap_for_plot = shap_values[1]
else:
    shap_for_plot = shap_values[:, :, 1]

summary_path = output_dir / "shap_summary_beeswarm.svg"
bar_path = output_dir / "shap_summary_bar.svg"
class_mean_path = output_dir / "shap_summary_class_mean_bars.svg"
class_mean_points_path = output_dir / "shap_summary_class_mean_points.svg"


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


shap.summary_plot(
    shap_for_plot,
    X_sample,
    plot_type="dot",
    cmap=SWARM_CMAP,
    plot_size=(figsize[0], figsize[1] * SWARM_HEIGHT_SCALE),
    max_display=SWARM_MAX_DISPLAY,
    show=False,
)
save_vector_svg(summary_path, fontsize=font_size, marker_size=marker_size)
print(f"Saved: {summary_path}")

shap.summary_plot(
    shap_for_plot,
    X_sample,
    plot_type="bar",
    plot_size=figsize,
    max_display=max_display,
    show=False,
)
save_vector_svg(bar_path, fontsize=font_size, marker_size=marker_size)
print(f"Saved: {bar_path}")

plot_class_mean_shap_bars(
    shap_for_plot,
    X_sample,
    X_sample.columns,
    max_display=max_display,
    figsize=figsize,
    cmap=cmap,
    fontsize=font_size,
    path=class_mean_path,
)
print(f"Saved: {class_mean_path}")

plot_class_mean_shap_points(
    shap_for_plot,
    X_sample,
    X_sample.columns,
    max_display=max_display,
    figsize=(figsize[0] * 0.85 * 0.85, figsize[1]),
    cmap=cmap,
    fontsize=font_size,
    path=class_mean_points_path,
)
print(f"Saved: {class_mean_points_path}")

print("SHAP version:", shap.__version__)
print("Expected value:", explainer.expected_value)

print(type(shap_values))

if isinstance(shap_values, list):
    print(shap_values[1].shape)
    print(shap_values[1].min(), shap_values[1].max())
else:
    print(shap_values.shape)
    print(shap_values[:, :, 1].min(), shap_values[:, :, 1].max())

print(f"\nFigures written to: {output_dir}")
print(
    f"figsize={figsize}, cmap={cmap}, max_display={max_display}, "
    f"font_size={font_size}, marker_size={marker_size}"
)
