import pandas as pd
from pathlib import Path
# CHANGED: file locations (main read and wrote everything in ~/Downloads)
REPO_DIR = Path(__file__).resolve().parent     # this repo
PLANTSHARED_DIR = REPO_DIR.parent / "plantshared_activatormodel"
YEAST_SHARED_DIR = REPO_DIR.parent / "yeast_shared1"
#combine the two files (plants only and shared). Add a new column to both files before concatenation -- plants only is 0 and shared is 1
#combined_yeastSpecificvsplantSpecific.py
plant_only = pd.read_csv(PLANTSHARED_DIR / "plantsOnlyActivatorsNOTyeast_finalData_june5.csv")
yeast_only = pd.read_csv(YEAST_SHARED_DIR / "y_activators.csv")



plant_only.columns = plant_only.columns.str.strip()
yeast_only.columns = yeast_only.columns.str.strip()



plant_only["label"] = 0
yeast_only["label"] = 1


combined = pd.concat([plant_only, yeast_only], ignore_index=True)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)


combined.to_csv(REPO_DIR / "June26YeastPlantSpecificcombined_activator_dataset1_finalData1.csv", index=False)
print(len(combined))
#combined = combined.drop(columns=[combined.columns[-2]])



print("Plant-only activators:", len(plant_only))
print("Yeast only activators:", len(yeast_only))
print("Total sequences:", len(combined))
