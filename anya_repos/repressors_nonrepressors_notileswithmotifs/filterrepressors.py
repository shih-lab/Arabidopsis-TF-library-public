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

SCRIPT_DIR = Path(__file__).resolve().parent
TF_LIBRARY_DATA = SCRIPT_DIR.parent.parent / "data"

INPUT_FILE = TF_LIBRARY_DATA / "plant_TF_all_data_annotated_features.csv"
REDUNDANT_DISORDER_COL = "Mean Disorder"

REPRESSORS_FILE = SCRIPT_DIR / "repressors_past.csv"
NONREPRESSORS_FILE = SCRIPT_DIR / "nonrepressors_past.csv"
FINAL_FILE = SCRIPT_DIR / "repressors_nonrepressors.csv"

FEATURE_COL = "#Feature"
QUALITY_COL = "quality"
SEQ_COL = "protein_seq"

RANDOM_STATE = 42

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


def has_known_rep_motif(seq):
    """True if protein_seq has at least one KNOW_REP_MOTIFS hit."""
    if not isinstance(seq, str):
        return False
    s = seq.strip().upper()
    if not s or s == "NAN":
        return False
    return any(pat.search(s) for pat in KNOW_REP_MOTIFS.values())


df = pd.read_csv(INPUT_FILE)

print("Original number of rows:", len(df))

df = df.drop_duplicates(subset=[FEATURE_COL], keep="first").copy()

print("Number of rows after removing duplicate #Feature values:", len(df))

if REDUNDANT_DISORDER_COL in df.columns:
    df = df.drop(columns=[REDUNDANT_DISORDER_COL])
    print(f"Dropped redundant column: {REDUNDANT_DISORDER_COL}")

if SEQ_COL not in df.columns:
    raise KeyError(
        f"Sequence column {SEQ_COL!r} not found in {INPUT_FILE}. "
        f"Columns: {list(df.columns[:20])}"
    )
has_motif = df[SEQ_COL].map(has_known_rep_motif)
print("Tiles with at least one canonical motif (removed):", int(has_motif.sum()))
df = df.loc[~has_motif].copy()
print("Number of rows after removing motif-containing tiles:", len(df))



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




nonrepressors = df[
    ~quality_normalized.isin(["repressor", "ambiguous repressor"])
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

ambiguous_count = (quality_normalized == "ambiguous repressor").sum()
print("Ambiguous repressors excluded:", ambiguous_count)

print("Final combined dataset:", len(final_df))

print("\nFinal label distribution:")
print(final_df["label"].value_counts())

print(f"\nFinal file saved to:\n{FINAL_FILE}")