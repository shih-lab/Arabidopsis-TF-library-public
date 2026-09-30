import pandas as pd
from pathlib import Path
#description: getting plants only activators compared against yeast file in order to do plants only vs. shared activator comparison

SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVATORS = SCRIPT_DIR / "activators_only_plantsdata_finaldata_june5.csv"
MORFFY = Path("/Users/JSAlamos/Documents/Shih_lab/TF_library/Morffy_all_data.xlsx")
CLEANED_OUT = SCRIPT_DIR / "activators_only_plantsdata_finaldata_june5_cleaned.csv"
PLANT_ONLY_OUT = SCRIPT_DIR / "plantsOnlyActivatorsNOTyeast_finalData_june5.csv"

activators = pd.read_csv(ACTIVATORS)
fragments = pd.read_excel(MORFFY)

#fragments = pd.read_excel("/Users/anyalachwani/Downloads/Morffy_all_data.xlsx")

# Remove rows with missing PADI Score
fragments = fragments.dropna(subset=["PADI Score"])

print("Fragments after removing NaN PADI rows:", len(fragments))


before_control = len(activators)

activators = activators[
    ~activators["#Feature"]
        .astype(str)
        .str.contains("control", case=False, na=False)
]

after_control = len(activators)

print("Rows before removing controls:", before_control)
print("Rows after removing controls:", after_control)
print("Control rows removed:", before_control - after_control)



# Create sets
feature_set = set(
    activators["#Feature"].astype(str).str.strip()
)

morffy_set = set(
    fragments["Fragment Name"].astype(str).str.strip()
)

# Keep only activators that exist in Morffy
activators = activators[
    activators["#Feature"]
        .astype(str)
        .str.strip()
        .isin(morffy_set)
]

print("Activators after removing non-Morffy features:", len(activators))
print("Unique activators remaining:",
      activators["#Feature"].nunique())

print("activators_length", len(activators))

activators.to_csv(CLEANED_OUT, index=False)

dups = activators["#Feature"].value_counts()
dups = dups[dups > 1]


print("Number of duplicated names:", len(dups))
print("Extra duplicate rows:", dups.sum() - len(dups))
print(dups.head(50))
feature_set = set(
    activators["#Feature"].astype(str).str.strip()
)

morffy_set = set(
    fragments["Fragment Name"].astype(str).str.strip()
)

print("Unique activators:", len(feature_set))
print("Unique Morffy names:", len(morffy_set))

missing = feature_set - morffy_set

print("\nFeatures present in activators but missing from Morffy:", len(missing))

for x in sorted(missing):
    print(x)

matched = fragments[
    fragments["Fragment Name"].astype(str).str.strip().isin(feature_set)
]



print("Matched rows:", len(matched))
print("Matched unique names:", matched["Fragment Name"].nunique())


nan_padi = fragments[
    (fragments["Fragment Name"].astype(str).str.strip().isin(feature_set))
    & (fragments["PADI Score"].isna())
]

print("\nMatched activators with NaN PADI:", len(nan_padi))

print(
    nan_padi[["Fragment Name", "PADI Score"]]
)

activators.columns = activators.columns.str.strip()
fragments.columns = fragments.columns.str.strip()


feature_set = set(activators["#Feature"].astype(str))


#filtered_fragments = fragments[
 #   (fragments["Fragment Name"].astype(str).isin(feature_set)) &
  #  (fragments["PADI Score"] <= 1)
#]


#filtered_fragments.to_csv("plantsOnlyActivatorsNOTyeast_finalData_june5.csv", index=False)

#print("Total fragments:", len(fragments))
#print("Matched rows with PADI < 1:", len(filtered_fragments))


padi_le1_names = set(
    fragments.loc[
        fragments["PADI Score"] <= 1,
        "Fragment Name"
    ].astype(str).str.strip()
)

plant_only_activators = activators[
    activators["#Feature"]
        .astype(str)
        .str.strip()
        .isin(padi_le1_names)
]

plant_only_activators.to_csv(PLANT_ONLY_OUT, index=False)

print("Plant-only activator rows:",
      len(plant_only_activators))
print("Plant-only unique activators:",
      plant_only_activators["#Feature"].nunique())
print(f"Wrote: {PLANT_ONLY_OUT}")
