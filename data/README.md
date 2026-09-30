# Input data (not included in this repository)

Pipeline scripts under `anya_repos/` expect annotated feature tables and run-specific CSVs in each project folder. Large files are omitted from GitHub; obtain them from your lab data store or regenerate them with the preprocessing scripts.

## Shared feature table

Several `annotatedfeaturesrun*.py` scripts read:

`data/plant_TF_all_data_annotated_features.csv`

Place that file here (same relative path as in the Shih lab `TF_library` layout). Jacob’s XGBoost notebooks in `jacob_notebooks/` use related tables such as `all_data_6_2026_relaxed_with_residual.csv` and `Anya_plant_TF_all_data_annotated_features.csv` in the notebook directory.

## Per-pipeline inputs

Each subdirectory under `anya_repos/` documents its run order in its `README.md`. Intermediate CSVs are produced by the filter scripts in that folder when starting from raw library exports.
