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


INPUT_FILE = "/Users/anyalachwani/Downloads/repressors_nonrepressors.csv"

OUTPUT_DIR = "/Users/anyalachwani/Downloads/repressor_batches_nonmutated"

LABEL_COL = "label"


N_REPRESSORS = 1745
N_NONREPRESSORS = 18684

RANDOM_STATE = 42

FEATURE_START_IDX = 17

TEST_SIZE = 0.30

TOP_N_SHAP = 100


os.makedirs(OUTPUT_DIR, exist_ok=True)



df = pd.read_csv(INPUT_FILE)
df = df.dropna(axis=1)


print("ORIGINAL DATASET")


print("Total rows:", len(df))

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())



repressors = df[df[LABEL_COL] == 1].copy()

nonrepressors = df[df[LABEL_COL] == 0].copy()
N_BATCHES = len(nonrepressors)//len(repressors)


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

    print(f"BATCH {batch_number:02d}")


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
        f"repressor_batch_{batch_number:02d}_nonmutated.csv"
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
        batch_df.columns[FEATURE_START_IDX:-1]
    )

    if LABEL_COL in feature_cols:
        feature_cols.remove(LABEL_COL)


    X = batch_df[feature_cols].copy()

    y = batch_df[LABEL_COL].copy()
    X = X.dropna(axis=1)
    feature_cols = X.columns.tolist()




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


    shap_values = explainer.shap_values(X_test)
    print("SHAP type:", type(shap_values))
    if isinstance(shap_values, list):
        shap_class1 = shap_values[1]
        print("SHAP shape:", shap_class1.shape)
        print("SHAP min/max:", shap_class1.min(), shap_class1.max())
    else:
        print("SHAP shape:", shap_values.shape)
        shap_class1 = shap_values[:, :, 1]
        print("SHAP min/max:", shap_class1.min(), shap_class1.max())




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
        f"repressor_batch_{batch_number:02d}_SHAP_nonmutated.csv"
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
        f"repressor_batch_{batch_number:02d}_SHAP_nonmutated.png"
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
    "nonrepressor_batch_usage_nonmutated.csv"
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
    "all_batches_SHAP_nonmutated.csv"
)


all_shap_df.to_csv(
    all_shap_file,
    index=False
)




