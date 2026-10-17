
#filter only overlapping activators (activators present in yeast and plant data)
#remove the rows where "#Feature" contains "control" in it
#updated filterData2activatorsoverlapping.py


import pandas as pd
from pathlib import Path
# CHANGED: file locations (main read and wrote everything in ~/Downloads)
REPO_DIR = Path(__file__).resolve().parent     # this repo
TF_LIBRARY_DIR = REPO_DIR.parent.parent        # TF_library folder (has data/ and the Morffy file)


activators = pd.read_csv(REPO_DIR / "activators_only_plantsdata_finaldata_june5.csv")
activators = activators.drop_duplicates(
    subset="#Feature",
    keep="first"
)
fragments = pd.read_excel(TF_LIBRARY_DIR / "Morffy_all_data.xlsx")

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

# Find matching fragment names with PADI >= 1
# CHANGED: >= 1 instead of > 1 (Morffy et al. call a fragment an activator at PADI >= 1)
matching_features = set(
    fragments.loc[
        fragments["PADI Score"] >= 1,
        "Fragment Name"
    ].astype(str)
)

# Keep rows from activators only
filtered_activators = activators[
    activators["#Feature"].astype(str).isin(matching_features)
]

filtered_activators.to_csv(
    REPO_DIR / "matched_activator_fragmentsSet2_actual_finalData_June5.csv",
    index=False
)

print("Total fragments:", len(fragments))
print("Matching activator fragments with PADI >= 1:", len(filtered_activators))

