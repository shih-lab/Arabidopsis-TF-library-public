# -*- coding: utf-8 -*-
"""
LxLxL ±2-flank pairwise-epistatic Ridge regressor (Spyder cell script).

Model summary
-------------
* Motif: first ``L.L.L`` match in each tile (strict EAR / LxLxL).
* Window: 5-mer core + ``flank`` residues on each side (default 9-mer ``xxLxLxLxx``).
* Fixed positions (the three leucines) are not encoded; all other positions are free.
* Features: one-hot main effects at each variable position **plus** one-hot
  pairwise products ``p{i}_p{j}`` (amino acid at i × amino acid at j) → epistasis.
* Outcome: ``z_log_enrichment_mean`` (plant tile repression enrichment).
* Fit: L2 Ridge (``alpha=50``), 75/25 train–test split (``random_state=1``).
* Calibration: linear map ``y ≈ cal_m * y_hat + cal_b`` fit on **train** predictions only.
* Logo: main-effect Ridge weights × ``cal_m`` (logomaker); pairwise terms omitted.

Spyder: run ``#%%`` cells top to bottom.

Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib, logomaker.

From ``scan_EAR_motifs.py``: run the launcher cell (``runpy.run_path``) or open this
file and run cells here.

Training data: ``Arabidopsis-TF-library-public/data/table S5_all_data.csv``.

Variables left for downstream cells in ``scan_EAR_motifs.py``:
``df``, ``window_col``, ``y_col``, ``motif_label``, ``model_epi``, ``enc``,
``cal_m``, ``cal_b``, ``X_cat``, ``extract_lxlxl_window``, ``make_feature_df_with_pairs``.
"""

#%% LxLxL epistatic Ridge — imports
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#%% LxLxL epistatic Ridge — settings
# --- Data ---
DEFAULT_INPUT_DATASET = Path(
    "/Users/JSAlamos/Documents/Shih_lab/TF_library/"
    "Arabidopsis-TF-library-public/data/table S5_all_data.csv"
)

# --- Motif geometry ---
flank = 2  # residues on each side of the 5-mer LxLxL core
PAD_CHAR = "-"  # pad when the ±flank window extends past the tile terminus
core_len = 5  # L . L . L (regex L.L.L)
core_pat = re.compile(r"L.L.L")  # first match only; middle two chars are wildcard
AA = list("ACDEFGHIKLMNPQRSTVWY")  # standard 20; padding "-" is a separate category in OHE

# --- Regression ---
y_col = "z_log_enrichment_mean"
sequence_col = "protein_seq"  # column name in Table S5
ridge_alpha = 50.0  # L2 strength (same as scan_EAR_motifs.py epistasis block)
test_size = 0.25
random_state = 1  # train/test split seed

# --- Derived window layout (0-based indices in the 9-mer) ---
# Example window:  p0 p1 L p3 L p5 L p7 p8
#                  ^flank^     core      ^flank^
window_len = core_len + 2 * flank
window_col = f"window{window_len}"  # column name on df, e.g. window9
motif_label = f"{'x' * flank}LxLxL{'x' * flank}"  # for plot labels only
fixed_L_pos = [flank, flank + 2, flank + 4]  # indices of the three leucines
var_pos = [p for p in range(window_len) if p not in fixed_L_pos]  # modeled positions

# logomaker color scheme (same as scan_EAR_motifs.py regression logos)
custom_colors = {
    "A": "darkgreen",
    "C": "mediumturquoise",
    "D": "mediumslateblue",
    "E": "mediumslateblue",
    "F": "darkgreen",
    "G": "goldenrod",
    "H": "orchid",
    "I": "darkgreen",
    "K": "orchid",
    "L": "darkgreen",
    "M": "darkgreen",
    "N": "mediumturquoise",
    "P": "goldenrod",
    "Q": "mediumturquoise",
    "R": "orchid",
    "S": "mediumturquoise",
    "T": "mediumturquoise",
    "V": "darkgreen",
    "W": "darkgreen",
    "Y": "mediumturquoise",
}

print("motif:", motif_label)
print("window_len:", window_len)
print("fixed_L_pos:", fixed_L_pos)
print("var_pos:", var_pos)

#%% LxLxL epistatic Ridge — helpers
def extract_lxlxl_window(seq):
    """
    Return the aligned motif window for one tile sequence, or None.

    Uses the first L.L.L hit. Edge tiles are padded with PAD_CHAR so every
    window has length window_len (no dropping of near-terminal motifs).
    """
    seq = str(seq).upper()
    m = core_pat.search(seq)
    if not m:
        return None

    s = m.start()
    left = s - flank
    right = s + core_len + flank

    # Pad/clamp when the symmetric window crosses tile boundaries
    left_pad = max(0, -left)
    right_pad = max(0, right - len(seq))
    left_clip = max(0, left)
    right_clip = min(len(seq), right)

    w = (
        PAD_CHAR * left_pad
        + seq[left_clip:right_clip]
        + PAD_CHAR * right_pad
    )

    if len(w) != window_len:
        return None
    # Regex allows I/V at degenerate sites; require leucine at all three fixed slots
    if not all(w[p] == "L" for p in fixed_L_pos):
        return None
    return w


def make_feature_df_with_pairs(series_window: pd.Series):
    """
    Build categorical design matrix before one-hot encoding.

    Columns p{k}: amino acid (or '-') at variable position k.
    Columns p{i}_p{j}: concatenated pair label 'A_K' for epistatic interaction terms.
    OneHotEncoder expands these into main + pairwise interaction coefficients in Ridge.
    """
    feat = pd.DataFrame(index=series_window.index)
    for p in var_pos:
        feat[f"p{p}"] = series_window.str[p]
    for i, j in combinations(var_pos, 2):
        feat[f"p{i}_p{j}"] = series_window.str[i] + "_" + series_window.str[j]
    return feat


#%% LxLxL epistatic Ridge — load Table S5
_input_path = DEFAULT_INPUT_DATASET
print("Loading:", _input_path)
_all_data = pd.read_csv(_input_path)

# Harmonize IDs / sequence column names with scan_EAR_motifs downstream cells
if "#Feature" in _all_data.columns and "tile_id" not in _all_data.columns:
    _all_data = _all_data.rename(columns={"#Feature": "tile_id"})
if sequence_col in _all_data.columns:
    _all_data["sequence"] = _all_data[sequence_col].astype(str).str.upper()

#%% LxLxL epistatic Ridge — extract windows → df
# Subset to tiles with a valid LxLxL window and finite phenotype
df = _all_data.copy()
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
df = df[np.isfinite(df[y_col])].copy()
df[sequence_col] = df[sequence_col].astype(str).str.upper()
if "sequence" not in df.columns:
    df["sequence"] = df[sequence_col]

df[window_col] = df[sequence_col].apply(extract_lxlxl_window)
df = df.dropna(subset=[window_col]).copy()

print(f"N tiles with {motif_label}:", len(df))

#%% LxLxL epistatic Ridge — fit train/test Ridge + train-only calibration
# Design: main effects + all pairwise interactions among var_pos (21 cat cols → ~2901 after OHE)
X_cat = make_feature_df_with_pairs(df[window_col])
y = df[y_col].values

train_idx, test_idx = train_test_split(
    df.index,
    test_size=test_size,
    random_state=random_state,
)

train_mask = df.index.isin(train_idx)
test_mask = df.index.isin(test_idx)

X_cat_train = X_cat.loc[train_idx]
X_cat_test = X_cat.loc[test_idx]
y_train = df.loc[train_idx, y_col].values
y_test = df.loc[test_idx, y_col].values

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train = enc.fit_transform(X_cat_train)
X_test = enc.transform(X_cat_test)

model_epi = Ridge(
    alpha=ridge_alpha,
    fit_intercept=True,
    random_state=0,
)
model_epi.fit(X_train, y_train)

yhat_train_raw = model_epi.predict(X_train)
yhat_test_raw = model_epi.predict(X_test)

# Map Ridge scores to enrichment scale using train set only (avoids test leakage)
cal_m, cal_b = np.polyfit(yhat_train_raw, y_train, 1)
yhat_train_cal = cal_m * yhat_train_raw + cal_b
yhat_test_cal = cal_m * yhat_test_raw + cal_b

df["y_pred_epi_raw"] = np.nan
df["y_pred_epi_cal"] = np.nan
df.loc[train_idx, "y_pred_epi_raw"] = yhat_train_raw
df.loc[test_idx, "y_pred_epi_raw"] = yhat_test_raw
df.loc[train_idx, "y_pred_epi_cal"] = yhat_train_cal
df.loc[test_idx, "y_pred_epi_cal"] = yhat_test_cal
df["split"] = np.where(train_mask, "train", "test")

print(f"N categorical features (pre one-hot): {X_cat.shape[1]}")
print(f"One-hot features: {enc.get_feature_names_out().shape[0]:,}")
print(f"\nCALIBRATION (train only)\nslope = {cal_m:.3f}\nintercept = {cal_b:.3f}")

for label, y_true, y_pred in (
    ("TRAIN RAW", y_train, yhat_train_raw),
    ("TEST RAW", y_test, yhat_test_raw),
    ("TRAIN CALIBRATED", y_train, yhat_train_cal),
    ("TEST CALIBRATED", y_test, yhat_test_cal),
):
    rho, p_rho = spearmanr(y_pred, y_true)
    r, p_r = pearsonr(y_pred, y_true)
    print(f"\n{label}")
    print(f"  Spearman rho = {rho:.3f} (p={p_rho:.2e})")
    print(f"  Pearson  r   = {r:.3f} (p={p_r:.2e})")

#%% LxLxL epistatic Ridge — calibrated prediction scatter (optional plot)
# Dashed line: pooled train+test linear trend (visual aid; calibration fit was train-only)
x_all = np.concatenate([yhat_train_cal, yhat_test_cal])
y_all = np.concatenate([y_train, y_test])
m, b = np.polyfit(x_all, y_all, 1)
xfit = np.linspace(x_all.min(), x_all.max(), 200)

plt.figure(figsize=(1.25, 1.25))
plt.scatter(
    yhat_train_cal,
    y_train,
    s=10,
    alpha=1,
    color="mediumseagreen",
    ec="none",
    label=f"train (r={pearsonr(yhat_train_cal, y_train).statistic:.2f})",
)
plt.scatter(
    yhat_test_cal,
    y_test,
    s=12,
    alpha=1,
    color="royalblue",
    ec="none",
    label=f"test (r={pearsonr(yhat_test_cal, y_test).statistic:.2f})",
)
plt.plot(xfit, m * xfit + b, color="r", linewidth=1, linestyle="--", zorder=2)
plt.xlabel("Calibrated prediction \n (log2 enrichment)")
plt.ylabel("Observation \n (log2 enrichment)")
plt.title("Pairwise epistasis model", fontsize=7)
plt.ylim(-4.3, 2.3)
plt.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()

#%% LxLxL epistatic Ridge — regression logo (main effects, calibrated weights)
# Coefficients from the epistatic Ridge model include pairwise terms; the logo
# shows only position-wise main effects (p0, p1, …), scaled by cal_m so heights
# are in calibrated log2-enrichment units. Sign flip: more negative = stronger
# predicted repression (stack height up = repression-promoting).
feature_names = enc.get_feature_names_out(X_cat.columns)
coef_s = pd.Series(model_epi.coef_, index=feature_names)

W = pd.DataFrame(index=AA, columns=[f"p{p}" for p in var_pos], dtype=float)
for p in var_pos:
    col = f"p{p}"
    v = pd.Series({aa: coef_s.get(f"{col}_{aa}", 0.0) for aa in AA})
    v = cal_m * v  # train calibration slope → enrichment scale
    v = v - v.mean()  # center within position (logo convention)
    W[col] = v

W_plot = -W.copy()
pos_map = {c: int(re.findall(r"\d+", str(c))[0]) for c in W_plot.columns}
logo_var = W_plot.rename(columns=pos_map).reindex(AA).T.copy()

logo_full = pd.DataFrame(0.0, index=list(range(window_len)), columns=AA)
logo_full.loc[logo_var.index, AA] = logo_var[AA].astype(float)

# Dry-run logo to fix y-limits before drawing fixed L glyphs at core positions
fig, ax = plt.subplots(figsize=(1.25, 1.25))
logomaker.Logo(logo_full, ax=ax, color_scheme=custom_colors, shade_below=0, fade_below=0)
ax.axhline(0, color="k", lw=0.8)
ymin, ymax = ax.get_ylim()
plt.close(fig)

core_height = ymax * 0.75 if ymax > 0 else 0.5
logo_full2 = logo_full.copy()
for p in fixed_L_pos:
    logo_full2.loc[p, "L"] = core_height

fig, ax = plt.subplots(figsize=(1.25, 1.25))
logomaker.Logo(
    logo_full2,
    ax=ax,
    color_scheme=custom_colors,
    shade_below=0,
    fade_below=0,
)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylim(ymin, ymax)
ax.set_ylabel("Repression contribution\n(calibrated log2 units)")
ax.set_xlabel(f"Position in {motif_label}")
ax.set_xticks(range(window_len))
ax.set_xticklabels(["x"] * window_len)
plt.title("Epistatic regression logo\n(main effects only)", fontsize=7)
plt.tight_layout()
plt.show()

#%% LxLxL epistatic Ridge — score EARlib1 (optional)
# Apply the train-fit model_epi + enc + (cal_m, cal_b) to an external peptide library.
_score_earlib = False  # set True to run this cell's scoring block

if _score_earlib:
    _earlib_path = (
        Path(__file__).resolve().parent / "EARlib1_sequences.csv"
    )
    lib = pd.read_csv(_earlib_path)
    lib["Sequence"] = lib["Sequence"].astype(str).str.upper()
    lib[window_col] = lib["Sequence"].apply(extract_lxlxl_window)
    lib["valid_EAR"] = lib[window_col].notna()

    valid_df = lib.loc[lib["valid_EAR"]].copy()
    X_cat_lib = make_feature_df_with_pairs(valid_df[window_col])
    X_lib = enc.transform(X_cat_lib)

    valid_df["predicted_strength_raw"] = model_epi.predict(X_lib)
    valid_df["predicted_strength_cal"] = (
        cal_m * valid_df["predicted_strength_raw"] + cal_b
    )

    lib["predicted_strength_raw"] = np.nan
    lib["predicted_strength_cal"] = np.nan
    lib.loc[valid_df.index, "predicted_strength_raw"] = valid_df[
        "predicted_strength_raw"
    ]
    lib.loc[valid_df.index, "predicted_strength_cal"] = valid_df[
        "predicted_strength_cal"
    ]

    # Lower calibrated score = stronger predicted repression (same sign convention as df)
    lib = lib.sort_values("predicted_strength_cal", ascending=True, na_position="last")
    outpath = str(_earlib_path).replace(".csv", f"_{motif_label}_predicted_calibrated.csv")
    lib.to_csv(outpath, index=False)
    print("Saved:", outpath)

#%% LxLxL epistatic Ridge — CLI (terminal only; skip in Spyder)
# Spyder: do not run this cell. Terminal batch export:
#   python ear_lxlxl_epistatic_ridge.py --output-predictions out.csv
def _cli_main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="LxLxL epistatic Ridge (batch mode).")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_DATASET)
    p.add_argument("--sequence-col", default="protein_seq")
    p.add_argument("--y-col", default="z_log_enrichment_mean")
    p.add_argument("--ridge-alpha", type=float, default=ridge_alpha)
    p.add_argument("--output-predictions", type=Path, default=None)
    args = p.parse_args()

    seq_col = args.sequence_col
    y_name = args.y_col
    alpha = args.ridge_alpha
    raw = pd.read_csv(args.input)
    if "#Feature" in raw.columns and "tile_id" not in raw.columns:
        raw = raw.rename(columns={"#Feature": "tile_id"})

    out = raw.copy()
    out[y_name] = pd.to_numeric(out[y_name], errors="coerce")
    out = out[np.isfinite(out[y_name])].copy()
    out[seq_col] = out[seq_col].astype(str).str.upper()
    out[window_col] = out[seq_col].apply(extract_lxlxl_window)
    out = out.dropna(subset=[window_col]).copy()

    x_cat = make_feature_df_with_pairs(out[window_col])
    tr_idx, _ = train_test_split(
        out.index, test_size=test_size, random_state=random_state
    )
    enc_cli = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_tr = enc_cli.fit_transform(x_cat.loc[tr_idx])
    mdl = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
    mdl.fit(x_tr, out.loc[tr_idx, y_name].values)
    yhat_tr = mdl.predict(x_tr)
    slope, intercept = np.polyfit(yhat_tr, out.loc[tr_idx, y_name].values, 1)
    out["y_pred_epi_raw"] = mdl.predict(enc_cli.transform(x_cat))
    out["y_pred_epi_cal"] = slope * out["y_pred_epi_raw"] + intercept

    if args.output_predictions:
        out.to_csv(args.output_predictions, index=False)
        print("Wrote", args.output_predictions)
    else:
        print(f"Fitted on {len(out):,} tiles; use --output-predictions to save.")


# Running the file as a script executes all cells above, then optionally the CLI.
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _cli_main()
