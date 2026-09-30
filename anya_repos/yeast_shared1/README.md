# yeast_shared1
yeast specific vs. shared

Order to run files
- `yeast_specificdata.py`
- `filter_finalData_yeastShared.py` (shared activators from `plantshared_activatormodel`)
- `annotatedfeaturesrunyeast.py`

SHAP figures are written to `shap_figures/` as vector SVGs. Style controls:

```bash
/Users/JSAlamos/miniconda3/envs/tf-shap/bin/python annotatedfeaturesrunyeast.py \
  --figsize 1.8 1.8 \
  --cmap coolwarm \
  --max-display 15 \
  --font-size 7 \
  --marker-size 5
```
