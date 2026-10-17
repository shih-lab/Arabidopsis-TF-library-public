#annotatedfeaturesrunrepressors.py
# Plant repressors (label 1) vs non-repressors (label 0), using the annotated features
# with no modifications, after removing every tile that has a known repressor motif
# (done in filterrepressors.py). The model has to find other sequence features.
#
# Batches are made as in run_repressors.py: every batch has all repressors plus an equal
# number of randomly chosen non-repressors, and the number of batches is
# (number of non-repressors) // (number of repressors). Lines marked "CHANGED:" differ from
# run_repressors.py. The main differences are:
#   1. Inside each batch, 5-fold cross-validation instead of one 70:30 train:test split.
#      All tiles from the same TF are kept in the same fold. Overlapping tiles of one TF are
#      very similar, so if some are in training and others in testing, the test score is too
#      optimistic. Every tile of the batch is tested exactly once.
#   2. Two random forests per fold, as in the activator models: the first one ranks the
#      features and the second one is trained on the top 100. Dropping NaN columns and picking
#      the top 100 is done inside each fold using only the training tiles.
#   3. Every tile in the dataset gets a repressor score: the mean probability of "repressor"
#      from all the fold models that never saw a tile from its TF (sequence_model_scores.csv).
#   4. Disorder features are included, with values from the latest annotated features file.
#   5. Results are averaged across batches, and figures are saved as SVG files (plotting.py).

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import shap

import plotting


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent
TF_LIBRARY_DIR = REPO_DIR.parent.parent

# Dataset made by filterrepressors.py
INPUT_FILE = REPO_DIR / "repressors_nonrepressors.csv"
# Annotated features of all plant tiles; used only to get the latest disorder values
SHARED_FEATURES_FILE = TF_LIBRARY_DIR / "data" / "plant_TF_all_data_annotated_features.csv"
OUTPUT_DIR = REPO_DIR / "shap_figures"

LABEL_COL = "label"
CLASS_NAMES = ("non-repressor", "repressor")  # label 0, label 1

RANDOM_STATE = 42

# CHANGED: 15 instead of 17, because "Mean Disorder" was dropped by filterrepressors.py
# and the disorder features are now included. Features start at tile_context_disorder.
FEATURE_START_IDX = 15

N_FOLDS = 5                   # 5-fold cross-validation inside each batch
N_TOP_FEATURES = 100          # features kept for the second random forest
N_SHAP_TILES_PER_FOLD = 200   # held-out tiles explained with SHAP in each fold
N_BEESWARM_TILES = 1000       # tiles drawn in the beeswarm plot (so the SVG stays small)

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

df = pd.read_csv(INPUT_FILE)

# CHANGED: take the two disorder features from the latest annotated features file
latest_disorder = pd.read_csv(
    SHARED_FEATURES_FILE,
    usecols=["#Feature", "tile_context_disorder", "tile_intrinsic_disorder"],
).drop_duplicates(subset=["#Feature"], keep="first")
latest_disorder = latest_disorder.set_index("#Feature")

for col in ["tile_context_disorder", "tile_intrinsic_disorder"]:
    new_values = df["#Feature"].map(latest_disorder[col])
    if col in df.columns:
        df[col] = new_values.fillna(df[col])  # keep the old value if a tile is missing
    else:
        df[col] = new_values

# Keep the label as the last column
df = df[[c for c in df.columns if c != LABEL_COL] + [LABEL_COL]]


print("ORIGINAL DATASET")

print("Total rows:", len(df))

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())


repressors = df[df[LABEL_COL] == 1].copy()

nonrepressors = df[df[LABEL_COL] == 0].copy()
N_BATCHES = len(nonrepressors) // len(repressors)

# Features are everything from FEATURE_START_IDX up to (not including) the label column
feature_cols = list(df.columns[FEATURE_START_IDX:-1])

print("\nRepressors available:", len(repressors))
print("Non-repressors available:", len(nonrepressors))
print("Number of features:", len(feature_cols))
print("First feature column:", feature_cols[0])


# CHANGED: group the tiles by TF, so all tiles of one TF end up in the same fold.
# Arabidopsis tile IDs start with the gene locus (e.g. AT1G01010_...). For other IDs we
# use everything before the last "_".
def get_tf_group(tile_id):
    match = re.match(r"^(AT[1-5CM]G\d{5})", tile_id, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return tile_id.rsplit("_", 1)[0]


all_groups = df["#Feature"].astype(str).apply(get_tf_group).to_numpy()

print("Number of TFs (groups):", len(np.unique(all_groups)))


print("\nBATCH SETTINGS")

print("Number of batches:", N_BATCHES)
print("Repressors per batch:", len(repressors))
print("Non-repressors per batch:", len(repressors))
print("Total rows per batch:", 2 * len(repressors))


# Results collected across batches
batch_cms = []                # confusion matrix of each batch
batch_metrics = []            # accuracy, precision/recall/F1 of each class, per batch
batch_roc_curves = []         # (true labels, predicted probabilities) per batch
batch_mean_abs_shap = []      # mean |SHAP| of each feature, per batch
all_shap_values = []          # SHAP values of the explained held-out tiles (all batches)
all_shap_tiles = []           # feature values of those tiles
fold_top_features = []        # top 100 features chosen in each fold (all batches)
fold_importances = []         # first-forest importance of every feature, per fold
tile_scores = []              # P(repressor) of every tile, from models that never saw its TF


for batch_number in range(1, N_BATCHES + 1):

    print(f"\n========== BATCH {batch_number:02d} ==========")

    seed = RANDOM_STATE + batch_number

    # All repressors (in random order) plus the same number of random non-repressors
    repressors_batch = repressors.sample(
        n=len(repressors),
        replace=False,
        random_state=seed
    )

    nonrepressors_batch = nonrepressors.sample(
        n=len(repressors),
        replace=False,
        random_state=seed
    )

    batch_df = pd.concat(
        [
            repressors_batch,
            nonrepressors_batch
        ],
        ignore_index=True
    )

    batch_df = batch_df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    print(
        f"Batch {batch_number:02d}: "
        f"{len(repressors_batch)} repressors + "
        f"{len(nonrepressors_batch)} non-repressors "
        f"= {len(batch_df)} total"
    )

    X = batch_df[feature_cols]
    y = batch_df[LABEL_COL]
    batch_groups = batch_df["#Feature"].astype(str).apply(get_tf_group)

    # Filled in fold by fold; every tile of the batch gets one out-of-fold prediction
    batch_predictions = np.full(len(batch_df), -1)
    batch_scores = np.full(len(batch_df), np.nan)   # predicted probability of "repressor"
    batch_shap_values = []

    # -----------------------------------------------------------------------
    # CHANGED: 5-fold cross-validation instead of one 70:30 train:test split.
    # StratifiedGroupKFold keeps all tiles of a TF in one fold and keeps the class ratio
    # similar across folds. Each fold: train on 4/5 of the TFs, test on the other 1/5.
    # -----------------------------------------------------------------------

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, batch_groups), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # CHANGED: columns with missing values are found in the training tiles only
        nan_cols = X_train.columns[X_train.isna().any()].tolist()
        X_train_no_nan = X_train.drop(columns=nan_cols)

        # First random forest: rank the features
        # (no class weights needed: the batch has the same number of tiles in each class)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1,
            random_state=seed
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

        # Keep only the top 100 features
        X_train_top = X_train[top_features]
        # A few test tiles have missing values in a selected feature. They are filled with
        # the median of the training tiles (or 0 if the whole training column is missing).
        train_medians = X_train_top.median()
        X_test_top = X_test[top_features].fillna(train_medians).fillna(0)

        # Second random forest: trained on the top 100 features only
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1,
            random_state=seed
        )

        model.fit(X_train_top, y_train)

        predictions = model.predict(X_test_top)
        scores = model.predict_proba(X_test_top)[:, 1]  # column 1 = probability of label 1

        print(f"Fold {fold}: {len(test_idx)} held-out tiles, "
              f"accuracy {accuracy_score(y_test, predictions):.3f}")

        batch_predictions[test_idx] = predictions
        batch_scores[test_idx] = scores
        fold_top_features.append(top_features)
        fold_importances.append(importance_df.set_index("Feature")["Importance"])

        # CHANGED: score every tile in the dataset whose TF was not in this fold's training
        # tiles (so no tile is scored by a model that saw its own TF)
        training_tfs = batch_groups.iloc[train_idx].unique()
        unseen = ~np.isin(all_groups, training_tfs)
        X_unseen = df.loc[unseen, top_features].fillna(train_medians).fillna(0)
        tile_scores.append(pd.DataFrame({
            "#Feature": df.loc[unseen, "#Feature"].to_numpy(),
            "batch": batch_number,
            "fold": fold,
            "score": model.predict_proba(X_unseen)[:, 1],
        }))

        # SHAP on a random sample of this fold's held-out tiles
        explainer = shap.TreeExplainer(model)

        X_sample = X_test_top.sample(
            n=min(N_SHAP_TILES_PER_FOLD, len(X_test_top)),
            random_state=seed
        )
        shap_values = explainer.shap_values(X_sample)

        # Binary classification: keep the SHAP values for label 1 (repressor)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[:, :, 1]

        # Each fold picks its own top 100, so we put the SHAP values in a table with all
        # features. Features this fold's model did not use get SHAP = 0 (no effect).
        shap_all_features = pd.DataFrame(0.0, index=X_sample.index, columns=feature_cols)
        shap_all_features[top_features] = shap_values
        batch_shap_values.append(shap_all_features)
        all_shap_tiles.append(X.loc[X_sample.index, feature_cols])

    # Performance of this batch: every tile was predicted by the model that did not see its TF
    cm = confusion_matrix(y, batch_predictions, labels=[0, 1])
    accuracy = accuracy_score(y, batch_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, batch_predictions, labels=[0, 1], zero_division=0
    )

    batch_cms.append(cm)
    batch_metrics.append([accuracy, precision[0], recall[0], f1[0], precision[1], recall[1], f1[1]])
    batch_roc_curves.append((y.to_numpy(), batch_scores))

    batch_shap_values = pd.concat(batch_shap_values).to_numpy()
    all_shap_values.append(batch_shap_values)
    batch_mean_abs_shap.append(
        pd.Series(np.abs(batch_shap_values).mean(axis=0), index=feature_cols)
    )

    print("Confusion matrix:\n", cm)
    print("Accuracy:", round(accuracy, 5))


# ---------------------------------------------------------------------------
# Results averaged across batches
# ---------------------------------------------------------------------------

mean_cm = np.mean(np.stack(batch_cms, axis=0), axis=0)
mean_metrics = np.mean(np.array(batch_metrics, dtype=float), axis=0)
mean_abs_shap = (
    pd.concat(batch_mean_abs_shap, axis=1)
    .mean(axis=1)
    .sort_values(ascending=False)
)

print("\n========== All batches ==========")
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


# How often each feature made the top 100 across all folds of all batches
selection_counts = pd.Series(0, index=feature_cols, dtype=int)
for top_features in fold_top_features:
    selection_counts[top_features] += 1

importance_table = pd.concat(fold_importances, axis=1)  # one column per fold
mean_importance = importance_table.reindex(feature_cols).fillna(0).mean(axis=1)

selection_df = pd.DataFrame({
    "feature": feature_cols,
    "n_folds_selected": selection_counts.values,
    "fraction_folds_selected": selection_counts.values / len(fold_top_features),
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


# CHANGED: repressor score of every tile = mean P(repressor) over the fold models that never
# saw its TF. Repressors are in every batch, so they get one score per batch; non-repressors
# also get scores from batches they were not part of.
print("\nSaving sequence scores (mean out-of-fold P(repressor))...")

score_info_cols = ["tf_id", "quality", "z_log_enrichment_mean", LABEL_COL]
tile_info = df.drop_duplicates(subset=["#Feature"])[["#Feature"] + score_info_cols]

tile_scores = pd.concat(tile_scores, ignore_index=True)
sequence_scores = (
    tile_scores.groupby("#Feature", as_index=False)
    .agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        n_models=("score", "size"),
        n_batches=("batch", "nunique"),
    )
    .merge(tile_info, on="#Feature", how="left")
    .sort_values([LABEL_COL, "mean_score"], ascending=[False, False])
    .reset_index(drop=True)
)
sequence_scores = sequence_scores[
    ["#Feature"] + score_info_cols + ["mean_score", "std_score", "n_models", "n_batches"]
]

scores_path = OUTPUT_DIR / "sequence_model_scores.csv"
sequence_scores.to_csv(scores_path, index=False)
print(f"Saved: {scores_path}")
print("Sequences scored:", len(sequence_scores))
print(sequence_scores.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Figures (saved as SVG in OUTPUT_DIR)
# ---------------------------------------------------------------------------

print("\nSaving performance figures (averaged across batches)...")

plotting.save_metrics_barplot_from_values(
    mean_metrics, OUTPUT_DIR / "performance_metrics_bar.svg",
    class_names=CLASS_NAMES,
    figsize=(max(FIGSIZE[0] * 2.5, 3.5), FIGSIZE[1]),  # wider, to fit the labels
    fontsize=FONT_SIZE,
)
plotting.save_truth_table_from_cm(
    mean_cm, OUTPUT_DIR / "truth_table.svg",
    class_names=CLASS_NAMES, figsize=FIGSIZE, fontsize=FONT_SIZE,
)
plotting.save_mean_roc_curve(
    batch_roc_curves, OUTPUT_DIR / "roc_curve.svg",
    figsize=FIGSIZE, fontsize=FONT_SIZE,
)

# SHAP
# SHAP values of the explained held-out tiles of all batches together (200 per fold).
# Features are sorted by their mean |SHAP| averaged across batches.
# Positive SHAP values (right side of the plots) push a tile toward "repressor".
feature_order = mean_abs_shap.index.tolist()
order_idx = [feature_cols.index(feature) for feature in feature_order]

shap_values = np.vstack(all_shap_values)[:, order_idx]
X_shap = pd.concat(all_shap_tiles, ignore_index=True)[feature_order]

print(f"\nSaving SHAP figures ({shap_values.shape[0]} tiles from {N_BATCHES} batches)...")

# The beeswarm draws one dot per tile and feature, so it only shows a random subset of tiles
rng = np.random.default_rng(RANDOM_STATE)
beeswarm_idx = np.sort(
    rng.choice(shap_values.shape[0], size=min(N_BEESWARM_TILES, shap_values.shape[0]), replace=False)
)

plotting.save_shap_beeswarm(
    shap_values[beeswarm_idx], X_shap.iloc[beeswarm_idx], OUTPUT_DIR / "shap_summary_beeswarm.svg",
    cmap=BEESWARM_CMAP, max_display=BEESWARM_MAX_DISPLAY,
    figsize=FIGSIZE, height_scale=BEESWARM_HEIGHT_SCALE,
    fontsize=FONT_SIZE, marker_size=MARKER_SIZE,
)
plotting.save_shap_bar(
    shap_values, X_shap, OUTPUT_DIR / "shap_summary_bar.svg",
    max_display=MAX_DISPLAY, figsize=FIGSIZE,
    fontsize=FONT_SIZE, marker_size=MARKER_SIZE,
)
plotting.save_class_mean_shap_bars(
    shap_values, X_shap, OUTPUT_DIR / "shap_summary_class_mean_bars.svg",
    max_display=MAX_DISPLAY, figsize=FIGSIZE, cmap=CMAP, fontsize=FONT_SIZE,
)
plotting.save_class_mean_shap_points(
    shap_values, X_shap, OUTPUT_DIR / "shap_summary_class_mean_points.svg",
    max_display=MAX_DISPLAY, figsize=(FIGSIZE[0] * 0.85 * 0.85, FIGSIZE[1]),
    cmap=CMAP, fontsize=FONT_SIZE,
)

print("SHAP version:", shap.__version__)
print(f"\nFigures written to: {OUTPUT_DIR}")
