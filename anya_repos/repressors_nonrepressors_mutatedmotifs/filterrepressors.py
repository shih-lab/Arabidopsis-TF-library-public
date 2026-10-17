#Repressor file

#plant repressor vs. plant non repressor build file

#take in one file, (csv) file1 
#remove the rows with duplicate #Feature values from the file, give total length after removing duplicates
#if quality = "repressor" --> filter to this and then save these rows in file called "repressors_new.csv" 

#if quality !="repressor" or "ambiguous repressor" save these rows in the file "nonrepressors.csv"

#combine "repressors_new.csv" and "nonrepressors.csv"

#add a new column "label" with 0 as the rows which were from the nonrepressors.csv file and with 1 for those in the rows from repressors_new.csv

#shuffle the rows of the entire file and save it to the file "repressors_final.csv"
 

#code

import pandas as pd
from pathlib import Path

# CHANGED: file locations (main read and wrote everything in ~/Downloads)
REPO_DIR = Path(__file__).resolve().parent     # this repo
TF_LIBRARY_DIR = REPO_DIR.parent.parent        # TF_library folder (has data/)


INPUT_FILE = TF_LIBRARY_DIR / "data" / "all_data_6_2026_relaxed_motif_mutated_annotated_features.csv"

REPRESSORS_FILE = REPO_DIR / "repressors_new.csv"
NONREPRESSORS_FILE = REPO_DIR / "nonrepressors.csv"
FINAL_FILE = REPO_DIR / "repressors_final__mutated.csv"

FEATURE_COL = "#Feature"
QUALITY_COL = "quality"

RANDOM_STATE = 42



df = pd.read_csv(INPUT_FILE)

# CHANGED: remove control tiles ("control_..."); they are not TF sequence
is_control = df[FEATURE_COL].astype(str).str.strip().str.lower().str.startswith("control_")
df = df[~is_control].copy()
print("Control tiles removed:", is_control.sum())

print("Original number of rows:", len(df))



df = df.drop_duplicates(subset=[FEATURE_COL], keep="first").copy()

print("Number of rows after removing duplicate #Feature values:", len(df))

# CHANGED: "Mean Disorder" is dropped because it is redundant with tile_context_disorder
if "Mean Disorder" in df.columns:
    df = df.drop(columns=["Mean Disorder"])



quality_normalized = (
    df[QUALITY_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)



repressors = df[quality_normalized == "repressor"].copy()

repressors.to_csv(REPRESSORS_FILE, index=False)

print("Number of repressors:", len(repressors))
print(f"Saved: {REPRESSORS_FILE}")




# Ambiguous repressors are kept as non-repressors. Main meant to exclude them, but it looked for
# "ambiguous repressor" (with a space) while the data has "ambiguous_repressor", so they were
# already kept as non-repressors; this line just says so explicitly.
nonrepressors = df[
    quality_normalized != "repressor"
].copy()

nonrepressors.to_csv(NONREPRESSORS_FILE, index=False)

print("Number of non-repressors:", len(nonrepressors))
print(f"Saved: {NONREPRESSORS_FILE}")




repressors["label"] = 1
nonrepressors["label"] = 0



final_df = pd.concat(
    [repressors, nonrepressors],
    ignore_index=True
)



final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)



final_df.to_csv(FINAL_FILE, index=False)



print("Rows after duplicate removal:", len(df))
print("Repressors (label 1):", len(repressors))
print("Non-repressors (label 0):", len(nonrepressors))

ambiguous_count = (quality_normalized == "ambiguous_repressor").sum()
print("Ambiguous repressors kept as non-repressors:", ambiguous_count)

print("Final combined dataset:", len(final_df))

print("\nFinal label distribution:")
print(final_df["label"].value_counts())

print(f"\nFinal file saved to:\n{FINAL_FILE}")
