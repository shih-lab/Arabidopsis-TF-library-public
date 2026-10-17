# Arabidopsis-TF-library-public

Public collection of Python pipelines and Jupyter notebooks for sequence-model and Random Forest / SHAP analyses on the Shih lab Arabidopsis TF library work (Anya's pipeline scripts and Jacob's XGBoost notebooks).

## Layout

| Path | Contents |
|------|----------|
| `anya_repos/` | Per-experiment Python pipelines (activator plant/shared, yeast-specific, repressor variants). Each folder has a `README.md` with run order. |
| `jacob_notebooks/` | XGBoost subset classifiers (`Anya_subset_xgb_*.ipynb`); notebook outputs cleared for version control. |
| `data/` | Place shared annotated feature CSVs here (see `data/README.md`). |
| `anya_repos/*/plotting.py` | Figure helpers imported by each folder's `annotatedfeaturesrun*.py` script (same file in every folder). |

Report: `anya_repos/ML_Sequence_Modeling_Report_August15.pdf`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the scripts in the order listed in each folder's README. Settings (input files, number of folds,
figure style) are constants at the top of each script. Example:

```bash
cd anya_repos/yeast_shared1
python annotatedfeaturesrunyeast.py
```

## Data

Large CSV / Excel inputs are **not** in this repo. See `data/README.md` and each subproject README for required filenames.

## Source

Assembled from local `Anya_repos` and `Jacob_notebooks` directories in the lab `TF_library` tree.
