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

import re

import pandas as pd
from pathlib import Path

# CHANGED: file locations (main read and wrote everything in ~/Downloads)
REPO_DIR = Path(__file__).resolve().parent     # this repo
TF_LIBRARY_DIR = REPO_DIR.parent.parent        # TF_library folder (has data/)


INPUT_FILE = TF_LIBRARY_DIR / "data" / "all_data_6_2026_relaxed_annotated_features.csv"

REPRESSORS_FILE = REPO_DIR / "repressors_past.csv"
NONREPRESSORS_FILE = REPO_DIR / "nonrepressors_past.csv"
FINAL_FILE = REPO_DIR / "repressors_nonrepressors.csv"

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

# CHANGED (this repo only): remove every tile with at least one known repressor motif,
# so the model has to find other sequence features. Motif panel (regular expressions):
# EAR motifs LxLxL, DLNxxP, DLNxP, DLNxxxP, and the non-EAR motifs WUS box, TIFY and TPL LFG.
KNOWN_REPRESSOR_MOTIFS = {
    "EAR_LxLxL": re.compile(r"L.L.L"),
    "EAR_DLNxxP": re.compile(r"DLN..P"),
    "EAR_DLNxP": re.compile(r"DLN.P"),
    "EAR_DLNxxxP": re.compile(r"DLN...P"),
    "WUS_box_TLxLFP": re.compile(r"TL.LFP(..)?"),
    "TIFY": re.compile(r"TIFY"),
    "TPL_LFG": re.compile(r"[RK]LFG[VI]"),
}


def has_known_repressor_motif(seq):
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    return any(motif.search(seq) for motif in KNOWN_REPRESSOR_MOTIFS.values())


has_motif = df["protein_seq"].map(has_known_repressor_motif)
df = df[~has_motif].copy()

print("Tiles with a known repressor motif (removed):", has_motif.sum())
print("Number of rows after removing motif tiles:", len(df))



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




# CHANGED: ambiguous repressors are kept as non-repressors (main excluded them from both classes)
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
