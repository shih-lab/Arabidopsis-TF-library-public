# repressors_nonrepressors_motifcount
Plant repressor vs. non-repressor classification using non-mutated annotated
features plus one extra feature: `n_known_rep_motifs`, the number of known
repressor-motif hits in `protein_seq` (0 if none).

Motifs are the `KNOW_REP_MOTIFS` panel (`EAR_MOTIFS | KNOWN_NON_EAR`):
LxLxL, DLNxxP, DLNxP, DLNxxxP, WUS box, TIFY, and TPL LFG.

Same input file as `repressors_nonrepressors_nomodifications`
(`all_data_6_2026_relaxed_annotated_features.csv`).

Order to run files
- `filterrepressors.py`
- `annotatedfeaturesrunrepressors.py`
- `run_repressors.py` (optional batch SHAP stability analysis)

`annotatedfeaturesrunrepressors.py` trains one model per non-repressor batch
(number of batches ≈ non-repressors / repressors). The truth table, metrics
barplot, and SHAP plots are **aggregated across batches** (mean CM/metrics;
SHAP values pooled over all batch test samples using a fixed top-feature set).
Each batch model also scores the full dataset; `shap_figures/sequence_model_scores.csv`
has every `#Feature` ID with mean / std `P(repressor)` across batches.

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
