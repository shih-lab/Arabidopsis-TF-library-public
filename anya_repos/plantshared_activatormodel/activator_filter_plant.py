
#add clause to include reading in the morffy set, see if that if the sequence's padi score in the second file is not given like it is nAN then remove that sequence's entire row from this file before saving it and give me the metrics. How many rows there were before and how many there are now.
#updated activator_filter_plant.py
import pandas as pd
from pathlib import Path
# CHANGED: file locations (main read and wrote everything in ~/Downloads)
REPO_DIR = Path(__file__).resolve().parent     # this repo
TF_LIBRARY_DIR = REPO_DIR.parent.parent        # TF_library folder (has data/ and the Morffy file)


df = pd.read_csv(TF_LIBRARY_DIR / "data" / "all_data_6_2026_relaxed_annotated_features.csv")
morffy = pd.read_excel(TF_LIBRARY_DIR / "Morffy_all_data.xlsx")

df.columns = df.columns.str.strip()
morffy.columns = morffy.columns.str.strip()


activators = df[
    df["quality"].astype(str).str.strip() == "activator"
].copy()

before_rows = len(activators)

nan_sequences = set(
    morffy.loc[
        morffy["PADI Score"].isna(),
        "Fragment Sequence"
    ]
    .astype(str)
    .str.strip()
)

activators = activators[
    ~activators["protein_seq"]
        .astype(str)
        .str.strip()
        .isin(nan_sequences)
].copy()

activators = activators.drop_duplicates(
    subset=["protein_seq"],
    keep="first"
).copy()


after_rows = len(activators)


activators.to_csv(
    REPO_DIR / "activators_only_plantsdata_finaldata_june5.csv",
    index=False
)
#check metrics
print("Plant activators before removing NaN PADI sequences:", before_rows)
print("Sequences with missing PADI Score in Morffy:", len(nan_sequences))
print("Plant activators after removing NaN PADI sequences:", after_rows)
print("Rows removed:", before_rows - after_rows)
