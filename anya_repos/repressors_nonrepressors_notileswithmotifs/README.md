# repressors_nonrepressors_notileswithmotifs
Plant repressor vs non-repressor model, using the annotated features with no modifications, after
removing every tile with at least one known repressor motif (LxLxL, DLNxxP, DLNxP, DLNxxxP, WUS box,
TIFY and TPL LFG).

Order of files run
- filterrepressors.py
- annotatedfeaturesrunrepressors.py

`run_repressors.py` is the original model from main (one 70:30 split per batch); it is kept for comparison.
`plotting.py` only holds the figure functions used by `annotatedfeaturesrunrepressors.py`. Figures,
`feature_selection_frequency.csv` (how many fold models picked each feature) and
`sequence_model_scores.csv` (repressor score of every tile) are saved in `shap_figures/`.

## What changed compared to main

Data preparation (`filterrepressors.py`, same as main except):
- File paths point to this folder and `TF_library/data` instead of `~/Downloads`.
- Control tiles (`control_...`) are removed.
- `Mean Disorder` is dropped because it is redundant with `tile_context_disorder`.
- Ambiguous repressors are kept as non-repressors (label 0); main excluded them from both classes.
- (this repo only) Tiles with a known repressor motif are removed here. The motifs are only used to
  remove tiles; they are not a model feature.

Model (`annotatedfeaturesrunrepressors.py`). Batches are made as in `run_repressors.py`: every batch has
all repressors plus the same number of randomly chosen non-repressors, and there are
(non-repressors // repressors) batches. What changed:
- 5-fold cross-validation inside each batch instead of one 70:30 train:test split. All tiles of the same
  TF are kept in the same fold, so a TF is never in training and testing at the same time (overlapping
  tiles of one TF are nearly identical, which made the 70:30 test score too optimistic). Every tile of
  the batch is tested exactly once.
- Two random forests per fold, as in the activator models: the first one ranks the features and the
  second one is trained on the top 100. Dropping NaN columns and choosing the top 100 features is done
  inside each fold, using only the training tiles.
- Every tile in the dataset gets a repressor score: the mean probability of "repressor" from all fold
  models that never saw a tile from its TF (`sequence_model_scores.csv`).
- The disorder features (`tile_context_disorder`, `tile_intrinsic_disorder`) are included, with values
  from the latest annotated features file.
- SHAP is computed on 200 held-out tiles per fold. Truth table, metrics and ROC curve are averaged
  across batches, and figures are saved as SVG files.
