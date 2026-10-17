# repressors_nonrepressors_nomodifications
Plant repressor vs. non-repressor classification using non-mutated annotated features.

Order to run files
- `filterrepressors.py`
- `annotatedfeaturesrunrepressors.py`
- `run_repressors.py` (optional batch SHAP stability analysis)

`annotatedfeaturesrunrepressors.py` builds class-balanced batches (all repressors
+ an equal random sample of non-repressors; number of batches = non-repressors //
repressors). Within each batch it runs 5-fold cross-validation grouped by TF locus
(`scripts/grouped_rf.py`), so overlapping tiles of one TF never straddle train and
test. NaN filtering and top-100 feature selection are redone inside every training
fold. The truth table, metrics barplot, and ROC curve use out-of-fold predictions,
averaged across batches; SHAP values are pooled over held-out tiles of every fold
(features a fold model did not select contribute SHAP = 0), and
`shap_figures/feature_selection_frequency.csv` reports how often each feature was
selected. `shap_figures/sequence_model_scores.csv` has every `#Feature` ID with the
mean / std `P(repressor)` over fold models whose training set had no tile from the
same TF (strictly out-of-fold).

SHAP figures are written to `shap_figures/` as vector SVGs. Style controls:

```bash
/Users/JSAlamos/miniconda3/envs/tf-shap/bin/python annotatedfeaturesrunrepressors.py \
  --figsize 1.8 1.8 \
  --cmap coolwarm \
  --max-display 15 \
  --font-size 7 \
  --marker-size 5 \
  --beeswarm-sample 1000
```
