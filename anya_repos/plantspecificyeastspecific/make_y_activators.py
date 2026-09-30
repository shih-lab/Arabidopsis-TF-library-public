"""Build yeast-specific activator table (y_activators.csv).

Yeast-specific = present in the plant library annotated features, yeast activator
in Morffy (PADI > 1), and not a plant activator.
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "plantshared_activatormodel"
ANNOTATED = Path(
    "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/"
    "plant_TF_all_data_annotated_features.csv"
)
MORFFY = Path("/Users/JSAlamos/Documents/Shih_lab/TF_library/Morffy_all_data.xlsx")
PLANT_ACTIVATORS = SHARED_DIR / "activators_only_plantsdata_finaldata_june5.csv"
OUT = SCRIPT_DIR / "y_activators.csv"

annotated = pd.read_csv(ANNOTATED)
morffy = pd.read_excel(MORFFY)
plant_acts = pd.read_csv(PLANT_ACTIVATORS)

annotated.columns = annotated.columns.str.strip()
morffy.columns = morffy.columns.str.strip()
plant_acts.columns = plant_acts.columns.str.strip()

yeast_names = set(
    morffy.loc[morffy["PADI Score"] > 1, "Fragment Name"]
    .astype(str)
    .str.strip()
)

plant_act_names = set(
    plant_acts["#Feature"]
    .astype(str)
    .str.strip()
)
plant_act_names = {n for n in plant_act_names if "control" not in n.lower()}

yeast_only_names = yeast_names - plant_act_names

y_activators = annotated[
    annotated["#Feature"].astype(str).str.strip().isin(yeast_only_names)
].copy()

y_activators = y_activators[
    ~y_activators["#Feature"]
    .astype(str)
    .str.contains("control", case=False, na=False)
]

y_activators.to_csv(OUT, index=False)

print("Yeast PADI>1 names:", len(yeast_names))
print("Plant activator names:", len(plant_act_names))
print("Yeast-only names:", len(yeast_only_names))
print("Yeast-only rows in annotated features:", len(y_activators))
print("Quality distribution:")
print(y_activators["quality"].value_counts())
print(f"Wrote: {OUT}")
