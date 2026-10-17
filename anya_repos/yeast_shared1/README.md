# yeast_shared1
yeast specific vs. shared

Order to run files
- yeast_specificdata.py
- filter_finalData_yeastShared.py (uses the shared activators made in `plantshared_activatormodel`)
- annotatedfeaturesrunyeast.py

`plotting.py` only holds the figure functions used by `annotatedfeaturesrunyeast.py`. Figures and
`feature_selection_frequency.csv` (how many of the 5 folds picked each feature) are saved in `shap_figures_yeast_vs_shared/`.

## What changed compared to main

Data preparation (same as main except):
- File paths point to this folder, `plantshared_activatormodel` and `TF_library/data` instead of `~/Downloads`.
- Yeast activators use PADI >= 1 (main: > 1), because Morffy et al. call a fragment an activator at
  PADI >= 1. No tile has PADI exactly 1, so the datasets don't change.

Model (`annotatedfeaturesrunyeast.py`). The model itself is the same: a first random forest ranks the
features and a second one is trained on the top 100, both with `class_weight="balanced"`. What changed:
- Control tiles (`control_...`) are removed.
- The class-0 subsampling step is removed (on main it kept all 1682 tiles anyway).
- The disorder features (`tile_context_disorder`, `tile_intrinsic_disorder`) are included, with values
  from the latest annotated features file. `Mean Disorder` is dropped because it is redundant.
- 5-fold cross-validation instead of one 60:40 train:test split. All tiles of the same TF are kept in the
  same fold, so a TF is never in training and testing at the same time (overlapping tiles of one TF are
  nearly identical, which made the 60:40 test score too optimistic). Every tile is tested exactly once.
- Dropping NaN columns and choosing the top 100 features is done inside each fold, using only the
  training tiles.
- SHAP is computed on 200 held-out tiles per fold (1000 in total), and figures are saved as SVG files.
