import pandas as pd
from pathlib import Path
# combine plant-only and yeast-only activators. plant-only is 0, yeast-only is 1

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "plantshared_activatormodel"
PLANT_ONLY = SHARED_DIR / "plantsOnlyActivatorsNOTyeast_finalData_june5.csv"
YEAST_ONLY = SCRIPT_DIR / "y_activators.csv"
OUT = SCRIPT_DIR / "June26YeastPlantSpecificcombined_activator_dataset1_finalData1.csv"

plant_only = pd.read_csv(PLANT_ONLY)
yeast_only = pd.read_csv(YEAST_ONLY)



plant_only.columns = plant_only.columns.str.strip()
yeast_only.columns = yeast_only.columns.str.strip()

DROP_COLS = ["tile_intrinsic_disorder"]

# Align yeast rows to the plant-only schema so column positions stay
# stable for the model script (features begin after col index 14).
# Missing plant metadata cols become NaN; yeast-only extras (e.g. PADI
# Score) are dropped.
yeast_only = yeast_only.reindex(columns=plant_only.columns)

plant_only = plant_only.drop(columns=[c for c in DROP_COLS if c in plant_only.columns])
yeast_only = yeast_only.drop(columns=[c for c in DROP_COLS if c in yeast_only.columns])

plant_only["label"] = 0
yeast_only["label"] = 1

combined = pd.concat([plant_only, yeast_only], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined.to_csv(OUT, index=False)
print(len(combined))
#combined = combined.drop(columns=[combined.columns[-2]])



print("Plant-only activators:", len(plant_only))
print("Yeast only activators:", len(yeast_only))
print("Total sequences:", len(combined))
print(f"Wrote: {OUT}")
