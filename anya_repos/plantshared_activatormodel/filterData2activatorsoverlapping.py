
#filter only overlapping activators (activators present in yeast and plant data)
#remove the rows where "#Feature" contains "control" in it

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVATORS = SCRIPT_DIR / "activators_only_plantsdata_finaldata_june5.csv"
MORFFY = Path("/Users/JSAlamos/Documents/Shih_lab/TF_library/Morffy_all_data.xlsx")
OUT = SCRIPT_DIR / "matched_activator_fragmentsSet2_actual_finalData_June5.csv"

activators = pd.read_csv(ACTIVATORS)
fragments = pd.read_excel(MORFFY)

activators = activators[
    ~activators["#Feature"]
        .astype(str)
        .str.contains("control", case=False, na=False)
].copy()


activators.columns = activators.columns.str.strip()
fragments.columns = fragments.columns.str.strip()


feature_set = set(activators["#Feature"].astype(str))


#filtered_fragments = fragments[
 #   (fragments["Fragment Name"].astype(str).isin(feature_set)) &
  #  (fragments["PADI Score"] > 1)
#]

# Find matching fragment names with PADI > 1
matching_features = set(
    fragments.loc[
        fragments["PADI Score"] > 1,
        "Fragment Name"
    ].astype(str)
)

# Keep rows from activators only
filtered_activators = activators[
    activators["#Feature"].astype(str).isin(matching_features)
]

filtered_activators.to_csv(OUT, index=False)

print("Total fragments:", len(fragments))
print("Matching activator fragments with PADI > 1:", len(filtered_activators))
print(f"Wrote: {OUT}")
