# plantshared_activatormodel
plant specific vs shared activator model

Order of files run
- activator_filter_plant.py
- filterData2activatorsoverlapping.py
- plantsonlyactivatorsNOTyeast.py
- filterToFinalData.py
- annotatedfeaturesrun.py

`plotting.py` only holds the figure functions used by `annotatedfeaturesrun.py`. Figures and
`feature_selection_frequency.csv` (how many of the 5 folds picked each feature) are saved in `shap_figures_plant_vs_shared/`.

## What changed compared to main

Data preparation (same as main except):
- File paths point to this folder and `TF_library/data` instead of `~/Downloads`.
- Shared activators use PADI >= 1 and plant-only activators use PADI < 1 (main: > 1 and <= 1),
  because Morffy et al. call a fragment an activator at PADI >= 1. No tile has PADI exactly 1, so the datasets don't change.

Model (`annotatedfeaturesrun.py`). The model itself is the same: a first random forest ranks the
features and a second one is trained on the top 100. What changed:
- Control tiles (`control_...`) are removed.
- No class-0 subsampling (on main it kept all 879 tiles anyway). Instead, both random forests use
  `class_weight="balanced"`.
- The disorder features (`tile_context_disorder`, `tile_intrinsic_disorder`) are included, with values
  from the latest annotated features file. `Mean Disorder` is dropped because it is redundant.
- 5-fold cross-validation instead of one 60:40 train:test split. All tiles of the same TF are kept in the
  same fold, so a TF is never in training and testing at the same time (overlapping tiles of one TF are
  nearly identical, which made the 60:40 test score too optimistic). Every tile is tested exactly once.
- Dropping NaN columns and choosing the top 100 features is done inside each fold, using only the
  training tiles.
- SHAP is computed on 200 held-out tiles per fold (1000 in total), and figures are saved as SVG files.
