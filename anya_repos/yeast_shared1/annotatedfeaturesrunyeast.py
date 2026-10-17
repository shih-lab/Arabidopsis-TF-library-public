#annotatedfeaturesrunyeast.py
# Yeast-only activators (label 0) vs shared plant + yeast activators (label 1).
#
# Same model as the original script on main (two random forests: the first one ranks
# the features, the second one is trained on the top 100). Lines marked "CHANGED:" differ
# from main. The main differences are:
#   1. Control tiles ("control_...") are removed; they are not TF sequence.
#   2. Instead of one 60:40 train:test split, we use 5-fold cross-validation where all
#      tiles from the same TF are kept in the same fold. Overlapping tiles of one TF are
#      very similar, so if some are in training and others in testing, the test score is
#      too optimistic. Every tile is tested exactly once, by the model that did not see its TF.
#   3. Dropping NaN columns and picking the top 100 features is done inside each fold using
#      only the training tiles, so the test tiles never influence which features are used.
#   4. Disorder features are included (main started the features after the disorder columns).
#   5. Figures are saved as SVG files instead of shown on screen (see plotting.py).

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap

import plotting


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent
TF_LIBRARY_DIR = REPO_DIR.parent.parent

# Dataset made by filter_finalData_yeastShared.py
INPUT_FILE = REPO_DIR / "June26yeast_combined_activator_dataset1_finalData1.csv"
# Annotated features of all plant tiles; used only to get the latest disorder values
SHARED_FEATURES_FILE = TF_LIBRARY_DIR / "data" / "plant_TF_all_data_annotated_features.csv"
# one folder name per model, so outputs don't overwrite each other if the activator
# scripts are run from the same folder
OUTPUT_DIR = REPO_DIR / "shap_figures_yeast_vs_shared"

CLASS_NAMES = ("yeast-only", "shared")  # label 0, label 1

N_FOLDS = 5                   # 5-fold cross-validation: each fold tests ~20% of the tiles
N_TOP_FEATURES = 100          # features kept for the second random forest
N_SHAP_TILES_PER_FOLD = 200   # held-out tiles explained with SHAP in each fold (1000 total)
RANDOM_STATE = 42

# Figure settings
FIGSIZE = (1.8, 1.8)          # inches
FONT_SIZE = 7
MARKER_SIZE = 5               # beeswarm dot size
BEESWARM_CMAP = "magma"
BEESWARM_MAX_DISPLAY = 25     # features shown in the beeswarm plot
BEESWARM_HEIGHT_SCALE = 1.75  # beeswarm is taller than the other plots
CMAP = "coolwarm"             # low/high feature value colors in the mean-SHAP plots
MAX_DISPLAY = 15              # features shown in the other SHAP plots

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plotting.apply_figure_style(fontsize=FONT_SIZE)


# ---------------------------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------------------------

dataset = pd.read_csv(INPUT_FILE)

print("Dataset shape:", dataset.shape)

# CHANGED: remove control tiles
is_control = (
    dataset["#Feature"].astype(str).str.strip().str.lower().str.startswith("control_")
)
dataset = dataset[~is_control].copy()
print("Control tiles removed:", is_control.sum())

# CHANGED: "Mean Disorder" is dropped because it is redundant with tile_context_disorder
if "Mean Disorder" in dataset.columns:
    dataset = dataset.drop(columns=["Mean Disorder"])

# CHANGED: take the two disorder features from the latest annotated features file
latest_disorder = pd.read_csv(
    SHARED_FEATURES_FILE,
    usecols=["#Feature", "tile_context_disorder", "tile_intrinsic_disorder"],
).drop_duplicates(subset=["#Feature"], keep="first")
latest_disorder = latest_disorder.set_index("#Feature")

for col in ["tile_context_disorder", "tile_intrinsic_disorder"]:
    new_values = dataset["#Feature"].map(latest_disorder[col])
    if col in dataset.columns:
        dataset[col] = new_values.fillna(dataset[col])  # keep the old value if a tile is missing
    else:
        dataset[col] = new_values

# Keep the label as the last column
label_col = "label"
dataset = dataset[[c for c in dataset.columns if c != label_col] + [label_col]]

print("Dataset shape after removing controls:", dataset.shape)

# CHANGED: the class-0 subsampling step was removed (on main it sampled all class-0
# tiles, so it did nothing). As on main, both random forests use class_weight="balanced".

# Shuffle
dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nLabel distribution:")
print(dataset[label_col].value_counts().sort_index())


# CHANGED: index 14 instead of 16, because "Mean Disorder" was dropped above and the
# disorder features are now included. Features start at index 15 (tile_context_disorder).
sequence_col_idx = 14
#one index before the start of the features

feature_cols = [c for c in dataset.columns[sequence_col_idx + 1:] if c != label_col]
#everything after sequence_col_idx except the label column

print("\nLast column before the features:", dataset.columns[sequence_col_idx])
print("Label column:", label_col)
print("Number of features:", len(feature_cols))

X = dataset[feature_cols]
y = dataset[label_col]


# CHANGED: group the tiles by TF, so all tiles of one TF end up in the same fold.
# Arabidopsis tile IDs start with the gene locus (e.g. AT1G01010_...). For other IDs we
# use everything before the last "_".
def get_tf_group(tile_id):
    match = re.match(r"^(AT[1-5CM]G\d{5})", tile_id, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return tile_id.rsplit("_", 1)[0]


groups = dataset["#Feature"].astype(str).apply(get_tf_group)

print("Number of TFs (groups):", groups.nunique())


# ---------------------------------------------------------------------------
# CHANGED: 5-fold cross-validation instead of one 60:40 train:test split.
# StratifiedGroupKFold keeps all tiles of a TF in one fold and keeps the class ratio
# similar across folds. Each fold: train on 4/5 of the TFs, test on the remaining 1/5.
# ---------------------------------------------------------------------------

cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Filled in fold by fold; every tile gets exactly one out-of-fold prediction
all_predictions = np.full(len(dataset), -1)
all_scores = np.full(len(dataset), np.nan)   # predicted probability of class 1
all_folds = np.zeros(len(dataset), dtype=int)

fold_top_features = []   # top 100 features chosen in each fold
fold_importances = []    # first-forest importance of every feature, per fold
shap_values_per_fold = []
shap_tiles_per_fold = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):

    print(f"\n========== Fold {fold} of {N_FOLDS} ==========")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("Training label distribution:")
    print(y_train.value_counts().sort_index())

    print("Testing label distribution:")
    print(y_test.value_counts().sort_index())

    # CHANGED: columns with missing values are found in the training tiles only
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    X_train_no_nan = X_train.drop(columns=nan_cols)

    print("Number of features after dropping NaN columns:", X_train_no_nan.shape[1])

    # First random forest: rank the features
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"  # larger class does not dominate
    )

    model.fit(X_train_no_nan, y_train)

    # Select Top 100 features using RF importance

    importance_df = pd.DataFrame({
        "Feature": X_train_no_nan.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    top_features = importance_df.head(N_TOP_FEATURES)["Feature"].tolist()

    print("\nTop 10 RF Features:")
    print(importance_df.head(10))

    # Keep only the top 100 features

    X_train_top = X_train[top_features]
    # CHANGED: a few test tiles have missing values in a selected feature. They are filled
    # with the median of the training tiles (or 0 if the whole training column is missing).
    X_test_top = X_test[top_features].fillna(X_train_top.median()).fillna(0)

    # Second random forest: trained on the top 100 features only
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"  # larger class does not dominate
    )

    model.fit(X_train_top, y_train)

    predictions = model.predict(X_test_top)
    scores = model.predict_proba(X_test_top)[:, 1]  # column 1 = probability of label 1

    print("\nFold accuracy:", accuracy_score(y_test, predictions))

    # Save this fold's results for the held-out tiles
    all_predictions[test_idx] = predictions
    all_scores[test_idx] = scores
    all_folds[test_idx] = fold
    fold_top_features.append(top_features)
    fold_importances.append(importance_df.set_index("Feature")["Importance"])

    # SHAP on a random sample of this fold's held-out tiles
    explainer = shap.TreeExplainer(model)

    X_sample = X_test_top.sample(
        n=min(N_SHAP_TILES_PER_FOLD, len(X_test_top)),
        random_state=42
    )
    shap_values = explainer.shap_values(X_sample)

    # Binary classification: keep the SHAP values for label 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    else:
        shap_values = shap_values[:, :, 1]

    # Each fold picks its own top 100, so we put the SHAP values in a table with all
    # features. Features this fold's model did not use get SHAP = 0 (no effect).
    shap_all_features = pd.DataFrame(0.0, index=X_sample.index, columns=feature_cols)
    shap_all_features[top_features] = shap_values
    shap_values_per_fold.append(shap_all_features)
    shap_tiles_per_fold.append(X.loc[X_sample.index, feature_cols])


# ---------------------------------------------------------------------------
# Results: all tiles, each predicted by the model that did not see its TF
# ---------------------------------------------------------------------------

y_test = y.to_numpy()
predictions = all_predictions
y_score = all_scores

print("\n========== All folds combined ==========")

for fold in range(1, N_FOLDS + 1):
    in_fold = all_folds == fold
    print(f"Fold {fold}: {in_fold.sum()} held-out tiles, "
          f"label counts {np.bincount(y_test[in_fold], minlength=2).tolist()}")

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# How often each feature made the top 100 across the 5 folds
selection_counts = pd.Series(0, index=feature_cols, dtype=int)
for top_features in fold_top_features:
    selection_counts[top_features] += 1

importance_table = pd.concat(fold_importances, axis=1)  # one column per fold
mean_importance = importance_table.reindex(feature_cols).fillna(0).mean(axis=1)

selection_df = pd.DataFrame({
    "feature": feature_cols,
    "n_folds_selected": selection_counts.values,
    "fraction_folds_selected": selection_counts.values / N_FOLDS,
    "mean_rf_importance": mean_importance.values,
})
selection_df = selection_df.sort_values(
    ["n_folds_selected", "mean_rf_importance"],
    ascending=False
).reset_index(drop=True)

selection_path = OUTPUT_DIR / "feature_selection_frequency.csv"
selection_df.to_csv(selection_path, index=False)
print(f"\nSaved: {selection_path}")
print(selection_df.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Figures (saved as SVG in OUTPUT_DIR)
# ---------------------------------------------------------------------------

print("\nSaving performance figures...")

plotting.save_metrics_barplot(
    y_test, predictions, OUTPUT_DIR / "performance_metrics_bar.svg",
    class_names=CLASS_NAMES,
    figsize=(max(FIGSIZE[0] * 2.5, 3.5), FIGSIZE[1]),  # wider, to fit the labels
    fontsize=FONT_SIZE,
)
plotting.save_truth_table(
    y_test, predictions, OUTPUT_DIR / "truth_table.svg",
    class_names=CLASS_NAMES, figsize=FIGSIZE, fontsize=FONT_SIZE,
)
plotting.save_roc_curve(
    y_test, y_score, OUTPUT_DIR / "roc_curve.svg",
    figsize=FIGSIZE, fontsize=FONT_SIZE,
)

# SHAP
print("\nSaving SHAP figures...")

# SHAP values of the 1000 explained held-out tiles (200 per fold), all folds together.
# Columns are sorted from most to least important (mean |SHAP|).
# Positive SHAP values (right side of the plots) push a tile toward label 1.
shap_values = pd.concat(shap_values_per_fold).to_numpy()
X_sample = pd.concat(shap_tiles_per_fold).reset_index(drop=True)

shap_order = np.argsort(-np.abs(shap_values).mean(axis=0))
shap_values = shap_values[:, shap_order]
X_sample = X_sample.iloc[:, shap_order]

plotting.save_shap_beeswarm(
    shap_values, X_sample, OUTPUT_DIR / "shap_summary_beeswarm.svg",
    cmap=BEESWARM_CMAP, max_display=BEESWARM_MAX_DISPLAY,
    figsize=FIGSIZE, height_scale=BEESWARM_HEIGHT_SCALE,
    fontsize=FONT_SIZE, marker_size=MARKER_SIZE,
)
plotting.save_shap_bar(
    shap_values, X_sample, OUTPUT_DIR / "shap_summary_bar.svg",
    max_display=MAX_DISPLAY, figsize=FIGSIZE,
    fontsize=FONT_SIZE, marker_size=MARKER_SIZE,
)
plotting.save_class_mean_shap_bars(
    shap_values, X_sample, OUTPUT_DIR / "shap_summary_class_mean_bars.svg",
    max_display=MAX_DISPLAY, figsize=FIGSIZE, cmap=CMAP, fontsize=FONT_SIZE,
)
plotting.save_class_mean_shap_points(
    shap_values, X_sample, OUTPUT_DIR / "shap_summary_class_mean_points.svg",
    max_display=MAX_DISPLAY, figsize=(FIGSIZE[0] * 0.85 * 0.85, FIGSIZE[1]),
    cmap=CMAP, fontsize=FONT_SIZE,
)

print("SHAP version:", shap.__version__)
print("SHAP matrix (explained tiles x features):", shap_values.shape)
print(f"\nFigures written to: {OUTPUT_DIR}")
