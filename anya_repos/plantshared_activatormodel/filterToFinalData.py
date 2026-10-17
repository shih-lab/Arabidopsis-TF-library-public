import pandas as pd
from pathlib import Path
# CHANGED: file locations (main read and wrote everything in ~/Downloads)
REPO_DIR = Path(__file__).resolve().parent     # this repo
TF_LIBRARY_DIR = REPO_DIR.parent.parent        # TF_library folder (has data/ and the Morffy file)
#combine the two files (plants only and shared). Add a new column to both files before concatenation -- plants only is 0 and shared is 1
# updated filterToFinalData.py
plant_only = pd.read_csv(REPO_DIR / "plantsOnlyActivatorsNOTyeast_finalData_june5.csv")
plant_yeast = pd.read_csv(REPO_DIR / "matched_activator_fragmentsSet2_actual_finalData_June5.csv")



plant_only.columns = plant_only.columns.str.strip()
plant_yeast.columns = plant_yeast.columns.str.strip()



plant_only["label"] = 0
plant_yeast["label"] = 1


combined = pd.concat([plant_only, plant_yeast], ignore_index=True)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)


combined.to_csv(REPO_DIR / "June26combined_activator_dataset1_finalData1.csv", index=False)
print(len(combined))
#combined = combined.drop(columns=[combined.columns[-2]])



print("Plant-only activators:", len(plant_only))
print("Plant+Yeast activators:", len(plant_yeast))
print("Total sequences:", len(combined))
