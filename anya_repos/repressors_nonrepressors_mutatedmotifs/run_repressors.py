import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


INPUT_FILE = "/Users/anyalachwani/Downloads/repressors_final__mutated.csv"

OUTPUT_DIR = "/Users/anyalachwani/Downloads/repressor_batches_mutated"

LABEL_COL = "label"

N_BATCHES = len(nonrepressors)/len(repressors)

N_REPRESSORS = 2000
N_NONREPRESSORS = 6000

RANDOM_STATE = 42

FEATURE_START_IDX = 21

TEST_SIZE = 0.20

TOP_N_SHAP = 100


os.makedirs(OUTPUT_DIR, exist_ok=True)



df = pd.read_csv(INPUT_FILE)


print("ORIGINAL DATASET")


print("Total rows:", len(df))

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())



repressors = df[df[LABEL_COL] == 1].copy()

nonrepressors = df[df[LABEL_COL] == 0].copy()

print("\nRepressors available:", len(repressors))
print("Non-repressors available:", len(nonrepressors))


n_repressors_to_use = min(
    N_REPRESSORS,
    len(repressors)
)


n_nonrepressors_to_use = min(
    n_repressors_to_use,
    len(nonrepressors)
)


print("BATCH SETTINGS")


print("Number of batches:", N_BATCHES)

print(
    "Repressors per batch:",
    n_repressors_to_use
)

print(
    "Non-repressors per batch:",
    n_nonrepressors_to_use
)

print(
    "Total rows per batch:",
    n_repressors_to_use
    + n_nonrepressors_to_use
)



nonrepressor_usage = pd.Series(
    0,
    index=nonrepressors.index,
    dtype=int
)


all_shap_results = []


for batch_number in range(1, N_BATCHES + 1):

    print("\n")
    print("=" * 60)
    print(f"BATCH {batch_number:02d}")
    print("=" * 60)

    seed = RANDOM_STATE + batch_number


    repressors_batch = repressors.sample(
        n=n_repressors_to_use,
        replace=False,
        random_state=seed
    ).copy()



    nonrepressors_batch = nonrepressors.sample(
        n=n_nonrepressors_to_use,
        replace=False,
        random_state=seed
    ).copy()



    nonrepressor_usage.loc[
        nonrepressors_batch.index
    ] += 1


   

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



    output_file = os.path.join(
        OUTPUT_DIR,
        f"repressor_batch_{batch_number:02d}.csv"
    )

    batch_df.to_csv(
        output_file,
        index=False
    )


    print(
        f"\nBatch {batch_number:02d}: "
        f"{len(repressors_batch)} repressors + "
        f"{len(nonrepressors_batch)} non-repressors "
        f"= {len(batch_df)} total"
    )



    feature_cols = list(
        batch_df.columns[FEATURE_START_IDX:]
    )

    if LABEL_COL in feature_cols:
        feature_cols.remove(LABEL_COL)


    X = batch_df[feature_cols].copy()

    y = batch_df[LABEL_COL].copy()


    X = X.fillna(
        X.median()
    )

    X = X.fillna(0)


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed
        )
    )


    print("\nTraining rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Features:", len(feature_cols))



    model = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )



    y_pred = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    print("\nMODEL PERFORMANCE")

    print(
        "Accuracy:",
        round(accuracy, 5)
    )

    print(
        "Precision:",
        round(precision, 5)
    )

    print(
        "Recall:",
        round(recall, 5)
    )

    print(
        "F1:",
        round(f1, 5)
    )


    print("\nCalculating SHAP")


    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        X_test
    )

    print("SHAP type:", type(shap_values))
    
    if isinstance(shap_values, np.ndarray):
        print("SHAP shape:", shap_values.shape)



    if isinstance(shap_values, list):

        shap_class1 = shap_values[1]

    elif (
        isinstance(shap_values, np.ndarray)
        and shap_values.ndim == 3
    ):

        shap_class1 = shap_values[:, :, 1]

    else:

        shap_class1 = shap_values



    mean_abs_shap = np.abs(
        shap_class1
    ).mean(axis=0)


    shap_importance = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Mean_ABS_SHAP": mean_abs_shap
        }
    )


    shap_importance = (
        shap_importance
        .sort_values(
            "Mean_ABS_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )


    shap_importance["Rank"] = (
        np.arange(
            1,
            len(shap_importance) + 1
        )
    )


    shap_importance["Batch"] = (
        batch_number
    )

    shap_csv_file = os.path.join(
        OUTPUT_DIR,
        f"repressor_batch_{batch_number:02d}_SHAP.csv"
    )


    shap_importance.to_csv(
        shap_csv_file,
        index=False
    )


    all_shap_results.append(
        shap_importance
    )



    plt.figure()


    shap.summary_plot(
        shap_class1,
        X_test,
        max_display=30,
        show=False
    )


    plt.title(
        f"Batch {batch_number:02d} - "
        "SHAP Summary"
    )


    plt.tight_layout()


    shap_plot_file = os.path.join(
        OUTPUT_DIR,
        f"repressor_batch_{batch_number:02d}_SHAP.png"
    )


    plt.savefig(
        shap_plot_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "SHAP plot saved:",
        shap_plot_file
    )



    print("\nTOP 20 SHAP FEATURES")

    print(
        shap_importance[
            [
                "Feature",
                "Mean_ABS_SHAP",
                "Rank"
            ]
        ]
        .head(20)
        .to_string()
    )



usage_df = nonrepressors.copy()

usage_df["number_of_batches_used"] = (
    nonrepressor_usage
)


usage_file = os.path.join(
    OUTPUT_DIR,
    "nonrepressor_batch_usage.csv"
)


usage_df.to_csv(
    usage_file,
    index=False
)


all_shap_df = pd.concat(
    all_shap_results,
    ignore_index=True
)


all_shap_file = os.path.join(
    OUTPUT_DIR,
    "all_batches_SHAP.csv"
)


all_shap_df.to_csv(
    all_shap_file,
    index=False
)




top_shap = all_shap_df[
    all_shap_df["Rank"] <= TOP_N_SHAP
].copy()


shap_stability = (
    top_shap
    .groupby("Feature")
    .agg(
        Top100_Batches=(
            "Batch",
            "nunique"
        ),

        Mean_SHAP=(
            "Mean_ABS_SHAP",
            "mean"
        ),

        Median_SHAP=(
            "Mean_ABS_SHAP",
            "median"
        ),

        Mean_Rank=(
            "Rank",
            "mean"
        ),

        Median_Rank=(
            "Rank",
            "median"
        )
    )
    .reset_index()
)




stability_file = os.path.join(
    OUTPUT_DIR,
    "SHAP_feature_stability.csv"
)


shap_stability.to_csv(
    stability_file,
    index=False
)





print("FINISHED")


print(
    f"\nCreated {N_BATCHES} batches."
)


print("\nFiles saved in:")

print(
    OUTPUT_DIR
)


print("\nEach batch contains approximately:")


print(
    f"{n_repressors_to_use} "
    "repressors (label 1)"
)


print(
    f"{n_nonrepressors_to_use} "
    "non-repressors (label 0)"
)


print(
    "\nNon-repressor sampling frequency:"
)


print(
    nonrepressor_usage
    .value_counts()
    .sort_index()
)


print(
    "\nAverage number of batches each "
    "non-repressor was used in:",
    round(
        nonrepressor_usage.mean(),
        2
    )
)




print("MOST STABLE SHAP FEATURES")



print(
    shap_stability[
        [
            "Feature",
            "Top100_Batches",
            "Mean_Rank",
            "Mean_SHAP"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


print("\nSHAP stability saved to:")

print(
    stability_file
)