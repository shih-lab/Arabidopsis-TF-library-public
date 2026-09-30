# repressors_nonrepressors_notileswithmotifs
Plant repressor vs. non-repressor classification using non-mutated annotated
features after removing every tile with at least one canonical repressor motif.

Motifs are the `KNOW_REP_MOTIFS` panel (`EAR_MOTIFS | KNOWN_NON_EAR`):
LxLxL, DLNxxP, DLNxP, DLNxxxP, WUS box, TIFY, and TPL LFG.

Same input feature file as `repressors_nonrepressors_nomodifications`
(`all_data_6_2026_relaxed_annotated_features.csv`). Motif presence is used
only to drop rows; it is not a model feature.

Order to run files
- `filterrepressors.py`
- `annotatedfeaturesrunrepressors.py`
- `run_repressors.py` (optional batch SHAP stability analysis)

`annotatedfeaturesrunrepressors.py` trains one model per non-repressor batch
(number of batches ≈ non-repressors / repressors). The truth table, metrics
barplot, and SHAP plots are **aggregated across batches** (mean CM/metrics;
SHAP values pooled over all batch test samples using a fixed top-feature set).
Each batch model also scores the full (motif-filtered) dataset;
`shap_figures/sequence_model_scores.csv` has every remaining `#Feature` ID
with mean / std `P(repressor)` across batches.

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
