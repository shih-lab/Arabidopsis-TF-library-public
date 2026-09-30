# plantspecificyeastspecific
plant specific activators vs. yeast specific activators

Order to run the files
- `make_y_activators.py` (uses Morffy + annotated features + plant activators from `plantshared_activatormodel`)
- `combined_yeastSpecificvsplantSpecific.py` (plant-only from `plantshared_activatormodel`)
- `annotatedfeaturesrunPlantSpecificYeastSpecific.py`

Plant-only inputs are produced by the `plantshared_activatormodel` pipeline.

SHAP figures are written to `shap_figures/` as vector SVGs. Style controls:

```bash
/Users/JSAlamos/miniconda3/envs/tf-shap/bin/python annotatedfeaturesrunPlantSpecificYeastSpecific.py \
  --figsize 1.8 1.8 \
  --cmap coolwarm \
  --max-display 15 \
  --font-size 7 \
  --marker-size 5
```
