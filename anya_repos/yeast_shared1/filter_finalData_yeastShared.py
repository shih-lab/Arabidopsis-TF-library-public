import pandas as pd
from pathlib import Path
# combine yeast-only and shared activators. yeast-only is 0, shared is 1
#filter_finalData_yeastShared.py

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "plantshared_activatormodel"
YEAST_ONLY = SCRIPT_DIR / "y_activators.csv"
PLANT_YEAST = SHARED_DIR / "matched_activator_fragmentsSet2_actual_finalData_June5.csv"
OUT = SCRIPT_DIR / "June26yeast_combined_activator_dataset1_finalData1.csv"

yeast_only = pd.read_csv(YEAST_ONLY)
plant_yeast = pd.read_csv(PLANT_YEAST)



yeast_only.columns = yeast_only.columns.str.strip()
plant_yeast.columns = plant_yeast.columns.str.strip()


yeast_only["label"] = 0
plant_yeast["label"] = 1


combined = pd.concat([yeast_only, plant_yeast], ignore_index=True)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)


combined.to_csv(OUT, index=False)
print(len(combined))
#combined = combined.drop(columns=[combined.columns[-2]])



print("Yeast-only activators:", len(yeast_only))
print("Plant+Yeast activators:", len(plant_yeast))
print("Total sequences:", len(combined))
print(f"Wrote: {OUT}")
