# plantspecificyeastspecific
plant specific activators vs. yeast specific activators

Order to run the files
- combined_yeastSpecificvsplantSpecific.py (Rest of the files can be obtained by running code in prior repos:
  plant-only activators from `plantshared_activatormodel`, `y_activators.csv` from `yeast_shared1`)
- annotatedfeaturesrunPlantSpecificYeastSpecific.py

`plotting.py` only holds the figure functions used by `annotatedfeaturesrunPlantSpecificYeastSpecific.py`.
Figures and `feature_selection_frequency.csv` (how many of the 5 folds picked each feature) are saved in `shap_figures_plantspecific_vs_yeastspecific/`.

## What changed compared to main

Data preparation (same as main except):
- File paths point to `plantshared_activatormodel` and `yeast_shared1` instead of `~/Downloads`.

Model (`annotatedfeaturesrunPlantSpecificYeastSpecific.py`). The model itself is the same: a first random
forest ranks the features and a second one is trained on the top 100, both with `class_weight="balanced"`.
What changed:
- Control tiles (`control_...`) are removed.
- The class-0 subsampling step is removed (on main it kept all 879 tiles anyway).
- The disorder features (`tile_context_disorder`, `tile_intrinsic_disorder`) are included, with values
  from the latest annotated features file. `Mean Disorder` is dropped because it is redundant.
- 5-fold cross-validation instead of one 60:40 train:test split. All tiles of the same TF are kept in the
  same fold, so a TF is never in training and testing at the same time (overlapping tiles of one TF are
  nearly identical, which made the 60:40 test score too optimistic). Every tile is tested exactly once.
- Dropping NaN columns and choosing the top 100 features is done inside each fold, using only the
  training tiles.
- SHAP is computed on 200 held-out tiles per fold (1000 in total), and figures are saved as SVG files.
