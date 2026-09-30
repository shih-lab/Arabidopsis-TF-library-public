import argparse
import re
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
from plot_classification_metrics import save_performance_figures, save_roc_curve

TF_LIBRARY_DATA = SCRIPT_DIR.parent.parent / "data"
SHARED_FEATURES_FILE = TF_LIBRARY_DATA / "plant_TF_all_data_annotated_features.csv"
DEFAULT_INPUT = SHARED_FEATURES_FILE
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "shap_figures"
CLASS_NAMES = ("non-repressor", "repressor")
LABEL_COL = "label"
ID_COL = "#Feature"
TF_ID_COL = "tf_id"
QUALITY_COL = "quality"
Z_COL = "z_log_enrichment_mean"
SEQUENCE_COL_IDX = 14
SEQ_COL = "protein_seq"
MOTIF_COUNT_COL = "n_known_rep_motifs"
N_REPRESSORS = 2000
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_N_FEATURES = 100
SWARM_MAX_DISPLAY = 25
SWARM_CMAP = "magma"
SWARM_HEIGHT_SCALE = 1.75
REDUNDANT_DISORDER_COL = "Mean Disorder"
EXCLUDE_FEATURE_COLS = {"PADI Score", "Mean Disorder"}
SCORE_CSV_NAME = "sequence_model_scores.csv"

# --- Configure motif regexes (same panel as scan_EAR_motifs.py) ---
EAR_MOTIFS = {
    "EAR_LxLxL_strict": re.compile(r"L.L.L"),
    "EAR_DLNxxP_strict": re.compile(r"DLN..P"),
    # Optional:
    "EAR_DLNxP_optional": re.compile(r"DLN.P"),
    "EAR_DLNxxxP_optional": re.compile(r"DLN...P"),
    # Optional hydrophobic broadened:
    # "EAR_hydrophobic_x3_optional": re.compile(r"[LIV].[LIV].[LIV]"),
}
# --- curated known non-EAR motif panel ---
KNOWN_NON_EAR = {
    "WUS_box_TLxLFP": re.compile(r"TL.LFP(..)?"),
    "TIFY_TIFY": re.compile(r"TIFY"),
    #"TIFY_TIxxYG": re.compile(r"TI..YG"),
    "TPL_LFG": re.compile(r"[RK]LFG[VI]"),
}

KNOW_REP_MOTIFS = EAR_MOTIFS | KNOWN_NON_EAR


def count_known_rep_motifs(seq):
    """Count KNOW_REP_MOTIFS hits in a protein sequence (0 if none)."""
    if not isinstance(seq, str):
        return 0
    s = seq.strip().upper()
    if not s or s == "NAN":
        return 0
    n = 0
    for pat in KNOW_REP_MOTIFS.values():
        n += sum(1 for _ in pat.finditer(s))
    return n


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
    class1_idx = list(model.classes_).index(1)
    y_score = model.predict_proba(X_test)[:, class1_idx]

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
    score_df = pd.DataFrame({
        ID_COL: full_df[ID_COL].values,
        TF_ID_COL: full_df[TF_ID_COL].values if TF_ID_COL in full_df.columns else np.nan,
        QUALITY_COL: full_df[QUALITY_COL].values if QUALITY_COL in full_df.columns else np.nan,
        Z_COL: full_df[Z_COL].values if Z_COL in full_df.columns else np.nan,
        LABEL_COL: full_df[LABEL_COL].values,
        MOTIF_COUNT_COL: full_df[MOTIF_COUNT_COL].values,
        "batch": batch_number,
        "score": proba[:, class1_idx],
    })

    return {
        "cm": cm,
        "metrics": metrics,
        "predictions": predictions,
        "y_test": y_test,
        "y_score": y_score,
        "X_sample": X_sample.reset_index(drop=True),
        "y_sample": y_sample.reset_index(drop=True),
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
        "Pass --input /path/to/plant_TF_all_data_annotated_features.csv"
    )

dataset = pd.read_csv(input_path)
print(f"Features source: {input_path}")
if REDUNDANT_DISORDER_COL in dataset.columns:
    dataset = dataset.drop(columns=[REDUNDANT_DISORDER_COL])
    print(f"Dropped redundant column: {REDUNDANT_DISORDER_COL}")
if LABEL_COL not in dataset.columns:
    quality_normalized = (
        dataset[QUALITY_COL].astype(str).str.strip().str.lower()
    )
    n_amb = int((quality_normalized == "ambiguous repressor").sum())
    dataset = dataset.loc[
        ~quality_normalized.isin(["ambiguous repressor"])
    ].copy()
    dataset[LABEL_COL] = (
        dataset[QUALITY_COL].astype(str).str.strip().str.lower() == "repressor"
    ).astype(int)
    print(f"Built labels from quality; excluded {n_amb} ambiguous repressors")
if SEQ_COL not in dataset.columns:
    raise KeyError(
        f"Sequence column {SEQ_COL!r} not found in {input_path}. "
        f"Columns: {list(dataset.columns[:20])}"
    )
dataset[MOTIF_COUNT_COL] = dataset[SEQ_COL].map(count_known_rep_motifs)
feature_cols_ordered = [
    col for col in dataset.columns if col not in (MOTIF_COUNT_COL, LABEL_COL)
] + [MOTIF_COUNT_COL, LABEL_COL]
dataset = dataset[feature_cols_ordered]

print("Dataset shape:", dataset.shape)
print("\nLabel distribution:")
print(dataset[LABEL_COL].value_counts().sort_index())
print(f"\n{MOTIF_COUNT_COL} distribution:")
print(dataset[MOTIF_COUNT_COL].value_counts().sort_index())
print(f"{MOTIF_COUNT_COL} by label (mean):")
print(dataset.groupby(LABEL_COL)[MOTIF_COUNT_COL].agg(["mean", "median", "max"]))

repressors = dataset[dataset[LABEL_COL] == 1].copy()
nonrepressors = dataset[dataset[LABEL_COL] == 0].copy()

n_repressors_to_use = min(N_REPRESSORS, len(repressors))
n_nonrepressors_to_use = min(n_repressors_to_use, len(nonrepressors))
n_batches = args.n_batches
if n_batches is None:
    n_batches = max(1, len(nonrepressors) // max(len(repressors), 1))

candidate_features = [
    c
    for c in dataset.columns[SEQUENCE_COL_IDX + 1:-1]
    if c not in EXCLUDE_FEATURE_COLS
]
if MOTIF_COUNT_COL not in candidate_features:
    candidate_features.append(MOTIF_COUNT_COL)

print("\nSelecting global top features for aligned SHAP across batches...")
top_features, dropped_nan_cols, importance_df = select_global_features(
    repressors, nonrepressors, candidate_features, TOP_N_FEATURES
)
if MOTIF_COUNT_COL not in top_features:
    top_features.append(MOTIF_COUNT_COL)
    print(f"Force-included {MOTIF_COUNT_COL} in model features")
print("Dropped NaN columns:", len(dropped_nan_cols))
print("Selected features:", len(top_features))
print(
    f"{MOTIF_COUNT_COL} RF rank:",
    int(importance_df["Feature"].tolist().index(MOTIF_COUNT_COL)) + 1
    if MOTIF_COUNT_COL in set(importance_df["Feature"])
    else "not ranked",
)
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
batch_scores = []
batch_roc = []
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
    batch_scores.append(result["score_df"])
    batch_roc.append((result["y_test"], result["y_score"]))
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
        n_known_rep_motifs=(MOTIF_COUNT_COL, "first"),
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
save_roc_curve(
    batch_curves=batch_roc,
    output_dir=output_dir,
    figsize=figsize,
    fontsize=font_size,
)

# Pool SHAP values from all batches; order features by batch-averaged mean |SHAP|.
feature_order = mean_abs_shap.index.tolist()
shap_for_plot = np.vstack(pooled_shap)
X_for_plot = pd.concat(pooled_X, ignore_index=True)[feature_order]
y_for_plot = pd.concat(pooled_y, ignore_index=True)
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

shap.summary_plot(
    shap_beeswarm,
    X_beeswarm,
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
    X_for_plot,
    X_for_plot.columns,
    max_display=max_display,
    figsize=figsize,
    cmap=cmap,
    fontsize=font_size,
    path=class_mean_path,
)
print(f"Saved: {class_mean_path}")

plot_class_mean_shap_points(
    shap_for_plot,
    X_for_plot,
    X_for_plot.columns,
    max_display=max_display,
    figsize=(figsize[0] * 0.85 * 0.85, figsize[1]),
    cmap=cmap,
    fontsize=font_size,
    path=class_mean_points_path,
)
print(f"Saved: {class_mean_points_path}")

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
