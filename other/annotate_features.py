#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate sequence datasets with compositional, biophysical, and grammar features.

Running the script with no arguments annotates the plant prelim library, Tycko
repressor/activator subtiles (40 aa, stride 10), Del Rosso subtiles, Ludwig 2023
hits subtiles, Erijman 2020 balanced AD subsample, Sanborn 2021 TF tiles (3x40 aa N/middle/C per 53 aa
fragment), and Markel repression peptides. Inputs must include sequences (e.g. prelim uses
`all_data_6-3-2026_coverage_filtered.csv` with Morffy `Fragment Sequence` and Twist
control DNA merged on `#Feature` into a `Sequence` column). Outputs end in
`_annotated_features.csv`.

Intrinsic disorder (`tile_intrinsic_disorder`) uses metapredict when installed.
`Mean Disorder` / `tile_context_disorder` are taken from Morffy_all_data.xlsx
(Fragment Name → Mean Disorder) when the feature id matches; otherwise they fall
back to metapredict intrinsic disorder. Run this script directly; do not stub
out metapredict before importing.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import metapredict as meta
except ImportError:
    meta = None


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VALID_AA = set(AMINO_ACIDS)

DIPEPTIDES_XX = [a + b for a, b in itertools.product(AMINO_ACIDS, repeat=2)]
DIPEPTIDES_XUX = [a + "_" + b for a, b in itertools.product(AMINO_ACIDS, repeat=2)]

AA_POSITIVE = set("KR")
AA_NEGATIVE = set("DE")
AA_CHARGED = AA_POSITIVE | AA_NEGATIVE

AA_POLAR = set("STNQY")
AA_NONPOLAR = set("AVILMFW")
AA_HYDROPHOBIC = set("AVILMFYW")
AA_ALIPHATIC = set("AVIL")
AA_AROMATIC = set("FWY")
AA_POLAR_UNCHARGED = set("NQTS")

# Residue groups for window-7 local clustering (M / mu).
AA_ALIPHATIC_MLIV = set("MLIV")
AA_POSITIVE_RK = set("RK")
AA_CHARGED_RKDE = AA_POSITIVE_RK | AA_NEGATIVE
WINDOW_CLUSTER_SIZE = 7
WINDOW_CLUSTERING_GROUPS = {
    "aliphatic": AA_ALIPHATIC_MLIV,
    "aromatic": AA_AROMATIC,
    "positive": AA_POSITIVE_RK,
    "negative": AA_NEGATIVE,
    "charged": AA_CHARGED_RKDE,
    "polar_uncharged": AA_POLAR_UNCHARGED,
}

AA_CLASSES = {
    "POS": AA_POSITIVE,
    "NEG": AA_NEGATIVE,
    "POL": AA_POLAR,
    "NONPOL": AA_NONPOLAR,
    "HYDRO": AA_HYDROPHOBIC,
    "ALIPH": AA_ALIPHATIC,
    "AROM": AA_AROMATIC,
}
CLASS_NAMES = list(AA_CLASSES.keys())
CLASS_PAIRS = [a + "-" + b for a, b in itertools.product(CLASS_NAMES, repeat=2)]

KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Chou-Fasman helix propensities (Pa values).
HELIX_PROPENSITY = {
    "A": 1.42,
    "C": 0.70,
    "D": 1.01,
    "E": 1.51,
    "F": 1.13,
    "G": 0.57,
    "H": 1.00,
    "I": 1.08,
    "K": 1.16,
    "L": 1.21,
    "M": 1.45,
    "N": 0.67,
    "P": 0.57,
    "Q": 1.11,
    "R": 0.98,
    "S": 0.77,
    "T": 0.83,
    "V": 1.06,
    "W": 1.08,
    "Y": 0.69,
}

# Eisenberg hydrophobicity scale (for amphipathic helix hydrophobic moment).
EISENBERG_HYDROPHOBICITY = {
    "A": 0.25,
    "R": -1.76,
    "N": -0.64,
    "D": -0.72,
    "C": 0.04,
    "Q": -0.69,
    "E": -0.62,
    "G": 0.16,
    "H": -0.40,
    "I": 0.73,
    "L": 0.53,
    "K": -1.10,
    "M": 0.26,
    "F": 0.61,
    "P": -0.07,
    "S": -0.26,
    "T": -0.18,
    "W": 0.37,
    "Y": 0.02,
    "V": 0.54,
}

DEFAULT_DATASETS = (
    {
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/"
            "all_data_6-3-2026_coverage_filtered.csv"
        ),
        "sequence_col": "Sequence",
        "output_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/"
            "all_data_6-3-2026_coverage_filtered_annotated_features.csv"
        ),
        "split_into_subtiles": False,
        "attach_reference_sequences": True,
        "feature_col": "#Feature",
        "drop_input_sequence_col": True,
    },
    {
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Tycko_2017/Tycko_repressors.xlsx"
        ),
        "sequence_col": "Extended Domain sequence",
        "output_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Tycko_2017/Tycko_repressors_subtiles40_stride10_annotated_features.csv"
        ),
        "split_into_subtiles": True,
    },
    {
        "id": "tycko_activators",
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Tycko_2017/Tycko_activators.xlsx"
        ),
        "sequence_col": "Extended Domain sequence",
        "output_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Tycko_2017/Tycko_activators_subtiles40_stride10_annotated_features.csv"
        ),
        "split_into_subtiles": True,
        "tycko_name_from_cols": ("Gene entry name", "tile_id"),
    },
    {
        "id": "delrosso_2023",
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "DelRosso_2023/Del_Rosso-supplement-Supplementary_Table_1_hitnohit_only.xlsx"
        ),
        "sequence_col": "Protein Sequence",
        "output_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "DelRosso_2023/"
            "Del_Rosso-supplement-Supplementary_Table_1_hitnohit_only_"
            "subtiles40_stride10_annotated_features.csv"
        ),
        "split_into_subtiles": True,
        "subtile_name_col": "label",
    },
    {
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Markel_ACS/eukaryotic_repression_peptides.csv"
        ),
        "sequence_col": "Domain",
        "output_path": None,
        "split_into_subtiles": False,
    },
    {
        "id": "ludwig_2023",
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Ludwig_2023/Ludwig_hits.csv"
        ),
        "sequence_col": "Max Sequence",
        "output_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Ludwig_2023/Ludwig_hits_subtiles40_stride10_annotated_features.csv"
        ),
        "split_into_subtiles": True,
        "subtile_name_col": "Tile ID",
    },
    {
        # All AD-positive plus matched random AD-negative subsample (~76k total).
        "id": "erijman_2020",
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Erijman_2020/Eirjman_compiled_data_1-s2.csv"
        ),
        "sequence_col": "sequence",
        "output_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Erijman_2020/"
            "Eirjman_compiled_data_1-s2_balanced_AD_annotated_features.csv"
        ),
        "subtile_mode": None,
        "balanced_subsample": {
            "label_col": "AD_set",
            "positive_label": "AD_positive",
            "negative_label": "AD_negative",
            "random_state": 0,
        },
    },
    {
        "id": "sanborn_2021",
        "input_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Sanborn_2021/Sanborn_fig1_data.csv"
        ),
        "sequence_col": "protein sequence",
        "output_path": Path(
            "/Users/JSAlamos/Documents/Shih_lab/TF_library/data/Other_peoples/"
            "Sanborn_2021/"
            "Sanborn_fig1_data_subtiles40_ncm_annotated_features.csv"
        ),
        "subtile_mode": "sanborn",
        "subtile_name_col": "subtile_id",
    },
)

MORFFY_ALL_DATA_PATH = Path(
    "/Users/JSAlamos/Documents/Shih_lab/TF_library/Morffy_all_data.xlsx"
)
TWIST_LIBRARY_PATH = Path(
    "/Users/JSAlamos/Documents/Shih_lab/TF_library/libraries/"
    "final_new_library_for_twist.csv"
)
TWIST_CODING_PREFIX_END = "GGTCTCTAGGT"
TWIST_CODING_SUFFIX_START = "GCTTAGAGACC"

_CODON_TABLE = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "_",
    "TAG": "_",
    "TGC": "C",
    "TGT": "C",
    "TGA": "_",
    "TGG": "W",
}

_WARNED_METAPREDICT = False
_WARNED_LOCALCIDER = False


def clean_sequence(seq) -> str:
    if pd.isna(seq):
        return ""
    return "".join(str(seq).split()).upper()


def blank_like_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        stripped = series.fillna("").astype(str).str.strip()
        mask = mask | stripped.eq("")
    return mask


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [col for col in df.columns if not blank_like_mask(df[col]).all()]
    return df.loc[:, keep_columns].copy()


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    blank_rows = pd.DataFrame({col: blank_like_mask(df[col]) for col in df.columns}).all(axis=1)
    return df.loc[~blank_rows].copy()


def validate_sequences(seqs: pd.Series) -> None:
    invalid_examples = []
    for idx, seq in seqs.items():
        invalid = sorted(set(seq) - VALID_AA)
        if invalid:
            invalid_examples.append((idx, "".join(invalid), seq))
        if len(invalid_examples) == 5:
            break

    if invalid_examples:
        details = "; ".join(
            f"row {idx}: invalid residues {bad!r} in sequence {seq!r}"
            for idx, bad, seq in invalid_examples
        )
        raise ValueError(f"Found non-canonical amino acids. {details}")


def _seq_len(seq: str) -> int:
    return len(seq) if seq else 0


def _fraction(seq: str, aa_set: set[str]) -> float:
    length = _seq_len(seq)
    if length == 0:
        return float("nan")
    return sum(aa in aa_set for aa in seq) / length


def fraction_positive(seqs):
    return [_fraction(seq, AA_POSITIVE) for seq in seqs]


def fraction_negative(seqs):
    return [_fraction(seq, AA_NEGATIVE) for seq in seqs]


def fraction_polar(seqs):
    return [_fraction(seq, AA_POLAR) for seq in seqs]


def fraction_nonpolar(seqs):
    return [_fraction(seq, AA_NONPOLAR) for seq in seqs]


def fraction_hydrophobic(seqs):
    return [_fraction(seq, AA_HYDROPHOBIC) for seq in seqs]


def fraction_aliphatic(seqs):
    return [_fraction(seq, AA_ALIPHATIC) for seq in seqs]


def fraction_aromatic(seqs):
    return [_fraction(seq, AA_AROMATIC) for seq in seqs]


def net_charge_density(pos_fracs, neg_fracs):
    """Net charge per residue (= frac_pos − frac_neg)."""
    out = []
    for pos, neg in zip(pos_fracs, neg_fracs):
        if math.isnan(pos) or math.isnan(neg):
            out.append(float("nan"))
        else:
            out.append(pos - neg)
    return out


def total_charged_fraction(pos_fracs, neg_fracs):
    """Total charged fraction (= frac_pos + frac_neg)."""
    out = []
    for pos, neg in zip(pos_fracs, neg_fracs):
        if math.isnan(pos) or math.isnan(neg):
            out.append(float("nan"))
        else:
            out.append(pos + neg)
    return out


def mean_hydrophobicity(seqs):
    out = []
    for seq in seqs:
        if not seq:
            out.append(float("nan"))
            continue
        out.append(sum(KYTE_DOOLITTLE[aa] for aa in seq) / len(seq))
    return out


def kappa_localcider(seqs, print_every=100):
    global _WARNED_LOCALCIDER

    try:
        from localcider.sequenceParameters import SequenceParameters
    except ImportError:
        if not _WARNED_LOCALCIDER:
            print("Warning: localcider is not installed; filling 'kappa' with NaN.")
            _WARNED_LOCALCIDER = True
        return [float("nan")] * len(seqs)

    out = []
    total = len(seqs)
    for i, seq in enumerate(seqs, start=1):
        if not seq:
            out.append(float("nan"))
        else:
            out.append(SequenceParameters(seq).get_kappa())
        if (i % print_every == 0) or (i == total):
            print(f"[kappa] processed {i}/{total} sequences")
    return out


def max_category_window_count(
    seq: str, aa_group: set[str], window: int = WINDOW_CLUSTER_SIZE
) -> float:
    """Max count of aa_group residues in any sliding window."""
    length = _seq_len(seq)
    if length == 0:
        return float("nan")
    if length < window:
        return float(sum(aa in aa_group for aa in seq))

    in_group = [aa in aa_group for aa in seq]
    count = sum(in_group[:window])
    best = count
    for i in range(window, length):
        if in_group[i - window]:
            count -= 1
        if in_group[i]:
            count += 1
        if count > best:
            best = count
    return float(best)


def category_window_M_over_mu(
    seq: str, aa_group: set[str], window: int = WINDOW_CLUSTER_SIZE
) -> float:
    """Peak window density vs composition-expected mean (M / mu); NaN if n=0."""
    length = _seq_len(seq)
    if length == 0:
        return float("nan")
    n = sum(aa in aa_group for aa in seq)
    if n == 0:
        return float("nan")
    m_peak = max_category_window_count(seq, aa_group, window=window)
    mu = window * n / length
    return m_peak / mu


def window_clustering_M_over_mu_features(
    seqs, window: int = WINDOW_CLUSTER_SIZE
) -> dict[str, list[float]]:
    out = {f"{name}_clustering": [] for name in WINDOW_CLUSTERING_GROUPS}
    for seq in seqs:
        for name, aa_group in WINDOW_CLUSTERING_GROUPS.items():
            out[f"{name}_clustering"].append(
                category_window_M_over_mu(seq, aa_group, window=window)
            )
    return out


def mean_helix_propensity(seqs):
    out = []
    for seq in seqs:
        if not seq:
            out.append(float("nan"))
            continue
        out.append(sum(HELIX_PROPENSITY[aa] for aa in seq) / len(seq))
    return out


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> tuple[float, float]:
    """Hydrophobic moment for an alpha helix (angle_deg ~100 for 3.6 residues/turn)."""
    if not seq:
        return float("nan"), float("nan")

    angle = np.deg2rad(angle_deg)
    x = 0.0
    y = 0.0
    vals = []
    for i, aa in enumerate(seq):
        hydro = EISENBERG_HYDROPHOBICITY.get(aa, 0.0)
        vals.append(hydro)
        x += hydro * np.cos(i * angle)
        y += hydro * np.sin(i * angle)

    moment = float(np.sqrt(x**2 + y**2) / len(seq))
    mean_hydro = float(np.mean(vals))
    return moment, mean_hydro


def max_hydrophobic_moment(
    seq: str,
    window: int = 18,
    angle_deg: float = 100.0,
) -> dict[str, float]:
    """Best amphipathic helix hydrophobic moment over sliding windows.

    max_hmoment_mean_hydro is the mean Eisenberg hydrophobicity of the
    18-residue window with the maximum hydrophobic moment (max_hmoment),
    not the window with maximum mean hydrophobicity.
    """
    if not seq:
        return {
            "max_hmoment": float("nan"),
            "max_hmoment_mean_hydro": float("nan"),
            "max_hmoment_start": float("nan"),
        }

    if len(seq) < window:
        moment, mean_hydro = hydrophobic_moment(seq, angle_deg=angle_deg)
        return {
            "max_hmoment": moment,
            # Mean Eisenberg hydrophobicity of the max-moment window.
            "max_hmoment_mean_hydro": mean_hydro,
            "max_hmoment_start": 0.0,
        }

    best = None
    for start in range(0, len(seq) - window + 1):
        window_seq = seq[start : start + window]
        moment, mean_hydro = hydrophobic_moment(window_seq, angle_deg=angle_deg)
        if best is None or moment > best["max_hmoment"]:
            best = {
                "max_hmoment": moment,
                # Mean Eisenberg hydrophobicity of the max-moment window.
                "max_hmoment_mean_hydro": mean_hydro,
                "max_hmoment_start": float(start),
            }
    return best


def max_hydrophobic_moment_features(
    seqs,
    window: int = 18,
    angle_deg: float = 100.0,
) -> dict[str, list[float]]:
    keys = ("max_hmoment", "max_hmoment_mean_hydro", "max_hmoment_start")
    out = {key: [] for key in keys}
    for seq in seqs:
        metrics = max_hydrophobic_moment(seq, window=window, angle_deg=angle_deg)
        for key in keys:
            out[key].append(metrics[key])
    return out


def shannon_entropy_aa(seqs):
    entropies = []
    for seq in seqs:
        length = _seq_len(seq)
        if length == 0:
            entropies.append(float("nan"))
            continue
        counts = Counter(seq)
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        entropies.append(entropy)
    return entropies


def shannon_entropy_aa_norm(seqs):
    max_entropy = math.log2(20)
    return [
        entropy / max_entropy if not math.isnan(entropy) else entropy
        for entropy in shannon_entropy_aa(seqs)
    ]


def fraction_all_aa(seqs):
    out = {aa: [] for aa in AMINO_ACIDS}
    for seq in seqs:
        length = len(seq)
        if length == 0:
            for aa in AMINO_ACIDS:
                out[aa].append(float("nan"))
            continue
        counts = Counter(seq)
        for aa in AMINO_ACIDS:
            out[aa].append(counts.get(aa, 0) / length)
    return out


def mean_intrinsic_disorder(fragment_seqs):
    global _WARNED_METAPREDICT

    if meta is None:
        if not _WARNED_METAPREDICT:
            print(
                "Warning: metapredict is not installed; filling "
                "'tile_intrinsic_disorder' with NaN."
            )
            _WARNED_METAPREDICT = True
        return [float("nan")] * len(fragment_seqs)

    out = []
    for seq in fragment_seqs:
        if not seq:
            out.append(float("nan"))
            continue
        scores = meta.predict_disorder(seq)
        out.append(float(np.mean(scores)))
    return out


def count_dipeptides_xx(seqs, normalize=False):
    out = {dp: [] for dp in DIPEPTIDES_XX}
    for seq in seqs:
        length = len(seq)
        if length < 2:
            for dp in DIPEPTIDES_XX:
                out[dp].append(0.0 if normalize else 0)
            continue
        counts = Counter(seq[i : i + 2] for i in range(length - 1))
        denom = (length - 1) if normalize else 1
        for dp in DIPEPTIDES_XX:
            out[dp].append(counts.get(dp, 0) / denom)
    return out


def count_dipeptides_xux(seqs, normalize=False):
    out = {dp: [] for dp in DIPEPTIDES_XUX}
    for seq in seqs:
        length = len(seq)
        if length < 3:
            for dp in DIPEPTIDES_XUX:
                out[dp].append(0.0 if normalize else 0)
            continue
        counts = Counter(seq[i] + "_" + seq[i + 2] for i in range(length - 2))
        denom = (length - 2) if normalize else 1
        for dp in DIPEPTIDES_XUX:
            out[dp].append(counts.get(dp, 0) / denom)
    return out


def _classes_for_residue(aa):
    return [name for name, aa_set in AA_CLASSES.items() if aa in aa_set]


def classpair_counts_xx(seqs, normalize=False):
    out = {cp: [] for cp in CLASS_PAIRS}
    for seq in seqs:
        length = len(seq)
        if length < 2:
            for cp in CLASS_PAIRS:
                out[cp].append(0.0 if normalize else 0)
            continue
        counts = Counter()
        for i in range(length - 1):
            left_classes = _classes_for_residue(seq[i])
            right_classes = _classes_for_residue(seq[i + 1])
            for left_class in left_classes:
                for right_class in right_classes:
                    counts[f"{left_class}-{right_class}"] += 1
        denom = (length - 1) if normalize else 1
        for cp in CLASS_PAIRS:
            out[cp].append(counts.get(cp, 0) / denom)
    return out


def classpair_counts_xux(seqs, normalize=False):
    out = {cp: [] for cp in CLASS_PAIRS}
    for seq in seqs:
        length = len(seq)
        if length < 3:
            for cp in CLASS_PAIRS:
                out[cp].append(0.0 if normalize else 0)
            continue
        counts = Counter()
        for i in range(length - 2):
            left_classes = _classes_for_residue(seq[i])
            right_classes = _classes_for_residue(seq[i + 2])
            for left_class in left_classes:
                for right_class in right_classes:
                    counts[f"{left_class}-{right_class}"] += 1
        denom = (length - 2) if normalize else 1
        for cp in CLASS_PAIRS:
            out[cp].append(counts.get(cp, 0) / denom)
    return out


def _max_run_one_gap(seq, aa):
    length = len(seq)
    if length == 0:
        return float("nan")
    left = 0
    mismatches = 0
    best = 0
    for right in range(length):
        if seq[right] != aa:
            mismatches += 1
        while mismatches > 1:
            if seq[left] != aa:
                mismatches -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


def longest_run_one_gap_all_aa(seqs):
    out = {aa: [] for aa in AMINO_ACIDS}
    for seq in seqs:
        for aa in AMINO_ACIDS:
            out[aa].append(_max_run_one_gap(seq, aa))
    return out


def feature_dict_to_frame(
    prefix: str,
    values: dict[str, list],
    index,
    key_transform=None,
) -> pd.DataFrame:
    if key_transform is None:
        key_transform = lambda key: key
    return pd.DataFrame(
        {f"{prefix}{key_transform(key)}": value for key, value in values.items()},
        index=index,
    )


def compute_feature_frame(raw_sequences: pd.Series) -> pd.DataFrame:
    seqs = raw_sequences.fillna("").map(clean_sequence)
    validate_sequences(seqs)
    index = raw_sequences.index

    intrinsic_disorder = mean_intrinsic_disorder(seqs)
    aa_features = feature_dict_to_frame("frac_", fraction_all_aa(seqs), index=index)
    frac_pos = fraction_positive(seqs)
    frac_neg = fraction_negative(seqs)
    summary_features = pd.DataFrame(
        {
            "frac_pos": frac_pos,
            "frac_neg": frac_neg,
            "frac_polar": fraction_polar(seqs),
            "frac_nonpolar": fraction_nonpolar(seqs),
            "frac_hydrophobic": fraction_hydrophobic(seqs),
            "frac_aliphatic": fraction_aliphatic(seqs),
            "frac_aromatic": fraction_aromatic(seqs),
            "frac_charged": total_charged_fraction(frac_pos, frac_neg),
            "charge_density": net_charge_density(frac_pos, frac_neg),
            "mean_hydrophobicity": mean_hydrophobicity(seqs),
            "tile_intrinsic_disorder": intrinsic_disorder,
            "kappa": kappa_localcider(seqs.tolist(), print_every=100),
            **window_clustering_M_over_mu_features(seqs.tolist()),
            "mean_helix_propensity": mean_helix_propensity(seqs),
            **max_hydrophobic_moment_features(seqs),
            "aa_entropy": shannon_entropy_aa_norm(seqs),
        },
        index=index,
    )
    xx_features = feature_dict_to_frame("XX_", count_dipeptides_xx(seqs, normalize=True), index=index)
    xux_features = feature_dict_to_frame(
        "XUX_",
        count_dipeptides_xux(seqs, normalize=True),
        index=index,
        key_transform=lambda key: key.replace("_", ""),
    )
    xx_class_features = feature_dict_to_frame(
        "XXcls_", classpair_counts_xx(seqs, normalize=True), index=index
    )
    xux_class_features = feature_dict_to_frame(
        "XUXcls_", classpair_counts_xux(seqs, normalize=True), index=index
    )
    run_features = feature_dict_to_frame("max_run1gap_", longest_run_one_gap_all_aa(seqs), index=index)

    return pd.concat(
        [
            aa_features,
            summary_features,
            xx_features,
            xux_features,
            xx_class_features,
            xux_class_features,
            run_features,
        ],
        axis=1,
    )


def balanced_label_subsample(
    df: pd.DataFrame,
    *,
    label_col: str,
    positive_label: str,
    negative_label: str,
    random_state: int = 0,
) -> pd.DataFrame:
    """Keep all positives and randomly subsample negatives to match."""
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} not found for balanced subsample.")

    positives = df.loc[df[label_col] == positive_label]
    negatives = df.loc[df[label_col] == negative_label]
    n_pos = len(positives)
    if n_pos == 0:
        raise ValueError(f"No rows with {label_col} == {positive_label!r}.")

    n_neg_available = len(negatives)
    n_neg_sample = min(n_neg_available, n_pos)
    neg_sample = negatives.sample(n=n_neg_sample, random_state=random_state)
    out = pd.concat([positives, neg_sample], ignore_index=True)
    print(
        f"Balanced subsample on {label_col!r}: "
        f"{n_pos:,} {positive_label!r}, "
        f"{n_neg_sample:,} {negative_label!r} "
        f"(sampled from {n_neg_available:,} negatives, seed={random_state})"
    )
    return out


def load_dataset(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(input_path, sep="\t")
    raise ValueError(f"Unsupported input format for {input_path}")


def strip_twist_overhangs(dna: str) -> str | None:
    """Return coding DNA between BsaI sites (after prefix end, before suffix start)."""
    normalized = clean_sequence(dna).replace("U", "T")
    prefix_index = normalized.find(TWIST_CODING_PREFIX_END)
    if prefix_index < 0:
        return None
    coding = normalized[prefix_index + len(TWIST_CODING_PREFIX_END) :]
    suffix_index = coding.find(TWIST_CODING_SUFFIX_START)
    if suffix_index < 0:
        return None
    return coding[:suffix_index]


_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


def translate_dna(seq: str) -> str | None:
    """Translate coding DNA to protein; stops at the first in-frame stop codon."""
    coding = clean_sequence(seq).replace("U", "T")
    if len(coding) % 3 != 0:
        return None
    residues = []
    for i in range(0, len(coding), 3):
        codon = coding[i : i + 3]
        if codon in _STOP_CODONS:
            break
        residues.append(_CODON_TABLE.get(codon, "X"))
    return "".join(residues)


def load_morffy_fragment_sequences(
    morffy_path: Path = MORFFY_ALL_DATA_PATH,
) -> pd.DataFrame:
    """Morffy tile_id -> protein sequence and Mean Disorder (40 aa fragments)."""
    morffy = pd.read_excel(
        morffy_path,
        usecols=["Fragment Name", "Fragment Sequence", "Mean Disorder"],
    )
    morffy = morffy.rename(
        columns={
            "Fragment Name": "feature_id",
            "Fragment Sequence": "sequence",
            "Mean Disorder": "mean_disorder",
        }
    )
    morffy["feature_id"] = morffy["feature_id"].astype(str).str.strip()
    morffy["sequence"] = morffy["sequence"].map(clean_sequence)
    morffy["mean_disorder"] = pd.to_numeric(morffy["mean_disorder"], errors="coerce")
    return morffy.drop_duplicates("feature_id", keep="first")


def load_control_sequences_from_twist(
    twist_path: Path = TWIST_LIBRARY_PATH,
) -> pd.DataFrame:
    """Control tile_id -> protein sequence from Twist library DNA (BsaI sites stripped)."""
    twist = pd.read_csv(twist_path, usecols=["Fragment Name", "DNA_BsaI_primers_overhangs"])
    controls = twist[
        twist["Fragment Name"].astype(str).str.contains("control_", case=False, na=False)
    ].copy()
    controls["feature_id"] = controls["Fragment Name"].astype(str).str.strip()
    controls["coding_dna"] = controls["DNA_BsaI_primers_overhangs"].map(strip_twist_overhangs)
    controls["sequence"] = controls["coding_dna"].map(translate_dna)
    controls = controls.drop_duplicates("feature_id", keep="first")
    missing = controls["sequence"].isna()
    if missing.any():
        names = controls.loc[missing, "feature_id"].tolist()
        raise ValueError(
            "Failed to extract or translate control coding DNA for: "
            f"{names[:10]}{'...' if len(names) > 10 else ''}"
        )
    return controls[["feature_id", "sequence"]]


def merge_reference_sequences(
    df: pd.DataFrame,
    *,
    feature_col: str = "#Feature",
    sequence_col: str = "sequence",
    morffy_path: Path = MORFFY_ALL_DATA_PATH,
    twist_path: Path = TWIST_LIBRARY_PATH,
) -> pd.DataFrame:
    """Fill ``sequence_col`` from Morffy fragments and Twist control DNA."""
    if feature_col not in df.columns:
        raise KeyError(f"Feature column {feature_col!r} was not found.")

    out = df.copy()
    if sequence_col not in out.columns:
        out[sequence_col] = pd.NA

    morffy = load_morffy_fragment_sequences(morffy_path)
    controls = load_control_sequences_from_twist(twist_path)
    morffy_map = morffy.set_index("feature_id")["sequence"]
    control_map = controls.set_index("feature_id")["sequence"]

    feature_ids = out[feature_col].astype(str).str.strip()
    is_control = feature_ids.str.contains("control_", case=False, na=False)
    n_blank_before = int(blank_like_mask(out[sequence_col]).sum())

    out.loc[is_control, sequence_col] = feature_ids.loc[is_control].map(control_map)
    needs_sequence = blank_like_mask(out[sequence_col])
    out.loc[needs_sequence, sequence_col] = feature_ids.loc[needs_sequence].map(
        morffy_map
    )

    n_filled = n_blank_before - int(blank_like_mask(out[sequence_col]).sum())
    n_controls = int(is_control.sum())
    n_still_missing = int(blank_like_mask(out[sequence_col]).sum())
    print(
        f"Attached reference sequences on {feature_col!r}: "
        f"updated {n_controls:,} control rows; "
        f"filled {n_filled:,} of {n_blank_before:,} blank rows; "
        f"{n_still_missing:,} still missing."
    )
    out = attach_morffy_mean_disorder(
        out, feature_col=feature_col, morffy_path=morffy_path, morffy=morffy
    )
    return out


def attach_morffy_mean_disorder(
    df: pd.DataFrame,
    *,
    feature_col: str = "#Feature",
    morffy_path: Path = MORFFY_ALL_DATA_PATH,
    morffy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Set ``Mean Disorder`` from Morffy where ``feature_col`` matches Fragment Name.

    Existing finite Mean Disorder values are kept; unmatched / missing rows stay
    NaN so callers can fall back to metapredict intrinsic disorder.
    """
    if feature_col not in df.columns:
        return df

    out = df.copy()
    if morffy is None:
        morffy = load_morffy_fragment_sequences(morffy_path)
    disorder_map = morffy.set_index("feature_id")["mean_disorder"]
    feature_ids = out[feature_col].astype(str).str.strip()
    from_morffy = feature_ids.map(disorder_map)

    if "Mean Disorder" in out.columns:
        existing = pd.to_numeric(out["Mean Disorder"], errors="coerce")
        # Prefer Morffy where matched; keep any pre-existing values only as fallback.
        out["Mean Disorder"] = from_morffy.fillna(existing)
    else:
        out["Mean Disorder"] = from_morffy

    n_morffy = int(from_morffy.notna().sum())
    n_final = int(pd.to_numeric(out["Mean Disorder"], errors="coerce").notna().sum())
    print(
        f"Attached Morffy Mean Disorder on {feature_col!r}: "
        f"{n_morffy:,} matched; {n_final:,} finite after merge."
    )
    return out


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_annotated_features.csv")


def build_tycko_subtiles(
    df: pd.DataFrame,
    sequence_col: str,
    *,
    window_size: int = 40,
    stride: int = 10,
) -> pd.DataFrame:
    rows = []
    for source_row_index, row in df.iterrows():
        original_sequence = clean_sequence(row[sequence_col])
        if not original_sequence or len(original_sequence) < window_size:
            continue

        # For 80aa Tycko tiles this yields 4 windows: starts 0, 10, 20, 30.
        last_start = len(original_sequence) - window_size - stride
        if last_start < 0:
            starts = [0]
        else:
            starts = list(range(0, last_start + 1, stride))

        for subtile_index, start in enumerate(starts, start=1):
            subtile_sequence = original_sequence[start : start + window_size]
            if len(subtile_sequence) != window_size:
                continue

            expanded_row = row.copy()
            expanded_row[sequence_col] = subtile_sequence
            expanded_row["tycko_source_row_index"] = source_row_index
            expanded_row["tycko_source_sequence"] = original_sequence
            expanded_row["tycko_subtile_index"] = subtile_index
            expanded_row["tycko_subtile_start_aa"] = start + 1
            expanded_row["tycko_subtile_end_aa"] = start + window_size
            rows.append(expanded_row)

    return pd.DataFrame(rows)


SANBORN_SUBTILE_POSITIONS = ("n_term", "middle", "c_term")


def build_sanborn_subtiles(
    df: pd.DataFrame,
    sequence_col: str,
    *,
    window_size: int = 40,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Extract N-terminal, middle, and C-terminal 40 aa windows from each row."""
    if id_col is None:
        for candidate in ("Unnamed: 0", "tile_id", "name", "ID"):
            if candidate in df.columns:
                id_col = candidate
                break

    rows = []
    for source_row_index, row in df.iterrows():
        original_sequence = clean_sequence(row[sequence_col])
        if not original_sequence or len(original_sequence) < window_size:
            continue

        fragment_length = len(original_sequence)
        position_starts = {
            "n_term": 0,
            "middle": (fragment_length - window_size) // 2,
            "c_term": fragment_length - window_size,
        }

        source_id = ""
        if id_col and id_col in row.index and not pd.isna(row[id_col]):
            source_id = str(row[id_col]).strip()

        for position_name in SANBORN_SUBTILE_POSITIONS:
            start = position_starts[position_name]
            subtile_sequence = original_sequence[start : start + window_size]
            if len(subtile_sequence) != window_size:
                continue

            expanded_row = row.copy()
            expanded_row[sequence_col] = subtile_sequence
            expanded_row["sanborn_source_row_index"] = source_row_index
            expanded_row["sanborn_source_sequence"] = original_sequence
            expanded_row["sanborn_subtile_position"] = position_name
            expanded_row["sanborn_subtile_start_aa"] = start + 1
            expanded_row["sanborn_subtile_end_aa"] = start + window_size
            if source_id:
                expanded_row["subtile_id"] = f"{source_id}_{position_name}"
            else:
                expanded_row["subtile_id"] = (
                    f"row{source_row_index}_{position_name}"
                )
            rows.append(expanded_row)

    return pd.DataFrame(rows)


def finalize_tycko_subtile_columns(
    df: pd.DataFrame,
    *,
    sequence_col: str,
    name_col: str = "Gene entry name",
    name_from_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Rename Tycko subtile outputs to standard `name` and `sequence` columns."""
    out = df.copy()
    if name_from_cols:
        missing = [col for col in name_from_cols if col not in out.columns]
        if missing:
            raise KeyError(f"Missing columns for Tycko subtile name: {missing}")
        if "name" in out.columns:
            out = out.drop(columns=["name"])
        out["name"] = out[list(name_from_cols)].astype(str).agg("".join, axis=1)
    elif name_col in out.columns and name_col != "name":
        if "name" in out.columns:
            out = out.drop(columns=["name"])
        out = out.rename(columns={name_col: "name"})
    if sequence_col in out.columns and sequence_col != "sequence":
        if "sequence" in out.columns:
            out = out.drop(columns=["sequence"])
        out = out.rename(columns={sequence_col: "sequence"})

    leading = [col for col in ("name", "sequence") if col in out.columns]
    if leading:
        remaining = [col for col in out.columns if col not in leading]
        out = out[leading + remaining]
    return out


def annotate_dataset(
    input_path: Path,
    sequence_col: str,
    output_path: Path | None = None,
    *,
    split_into_subtiles: bool = False,
    subtile_mode: str | None = None,
    subtile_name_col: str = "Gene entry name",
    tycko_name_from_cols: tuple[str, ...] | None = None,
    row_filter: str | None = None,
    balanced_subsample: dict[str, str | int] | None = None,
    attach_reference_sequences: bool = False,
    feature_col: str = "#Feature",
    drop_input_sequence_col: bool = False,
) -> Path:
    df = drop_empty_columns(drop_empty_rows(load_dataset(input_path)))
    if attach_reference_sequences:
        df = merge_reference_sequences(
            df, feature_col=feature_col, sequence_col=sequence_col
        )
        if (
            drop_input_sequence_col
            and sequence_col != "sequence"
            and "sequence" in df.columns
        ):
            df = df.drop(columns=["sequence"])
    if balanced_subsample:
        df = balanced_label_subsample(df, **balanced_subsample)
    elif row_filter:
        before = len(df)
        df = df.query(row_filter, engine="python").copy()
        print(f"Applied row filter ({row_filter!r}): {before:,} -> {len(df):,} rows")
        if df.empty:
            raise ValueError(f"Row filter {row_filter!r} matched no rows in {input_path}.")
    if sequence_col not in df.columns:
        available = ", ".join(df.columns.astype(str))
        raise KeyError(
            f"Sequence column {sequence_col!r} was not found in {input_path}. "
            f"Available columns: {available}"
        )

    df = df.copy()
    cleaned_sequences = df[sequence_col].map(clean_sequence)
    df = df.loc[cleaned_sequences.ne("")].copy()

    resolved_subtile_mode = subtile_mode
    if resolved_subtile_mode is None and split_into_subtiles:
        resolved_subtile_mode = "tycko"

    if resolved_subtile_mode == "tycko":
        df = build_tycko_subtiles(df, sequence_col=sequence_col)
        if df.empty:
            raise ValueError(
                "No valid Tycko subtiles were generated. Check sequence length/content."
            )
    elif resolved_subtile_mode == "sanborn":
        df = build_sanborn_subtiles(df, sequence_col=sequence_col)
        if df.empty:
            raise ValueError(
                "No valid Sanborn subtiles were generated. Check sequence length/content."
            )
    elif resolved_subtile_mode is not None:
        raise ValueError(f"Unknown subtile_mode: {resolved_subtile_mode!r}")

    feature_frame = compute_feature_frame(df[sequence_col])
    intrinsic_disorder = feature_frame["tile_intrinsic_disorder"]

    # Prefer Morffy Mean Disorder (context) when feature ids match Fragment Name.
    if feature_col in df.columns:
        df = attach_morffy_mean_disorder(df, feature_col=feature_col)

    if "Mean Disorder" in df.columns:
        existing_mean_disorder = pd.to_numeric(df["Mean Disorder"], errors="coerce")
        df["Mean Disorder"] = existing_mean_disorder.fillna(intrinsic_disorder)
    else:
        df["Mean Disorder"] = intrinsic_disorder

    # Context disorder is Morffy's Mean Disorder (not metapredict intrinsic).
    df["tile_context_disorder"] = df["Mean Disorder"]

    annotated = pd.concat([df, feature_frame], axis=1)
    if resolved_subtile_mode is not None:
        annotated = finalize_tycko_subtile_columns(
            annotated,
            sequence_col=sequence_col,
            name_col=subtile_name_col,
            name_from_cols=tycko_name_from_cols,
        )
    resolved_output = output_path or build_output_path(input_path)
    annotated.to_csv(resolved_output, index=False)
    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate sequence tables with biochemical and grammar features. "
            "If no arguments are provided, the Tycko and Markel datasets are annotated."
        )
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help="Input dataset path. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--sequence-col",
        dest="sequence_cols",
        action="append",
        help="Sequence column for each input dataset. Repeat in the same order as --input.",
    )
    parser.add_argument(
        "--output",
        dest="outputs",
        action="append",
        help="Optional output CSV path for each input dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="ID",
        help=(
            "When using built-in defaults (no --input), run only these dataset "
            "ids (e.g. ludwig_2023)."
        ),
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict[str, Path | str | None]]:
    if not args.inputs:
        datasets = DEFAULT_DATASETS
        if args.datasets:
            selected = set(args.datasets)
            datasets = [d for d in datasets if d.get("id") in selected]
            missing = selected - {d.get("id") for d in datasets}
            if missing:
                known = sorted(
                    {d["id"] for d in DEFAULT_DATASETS if d.get("id")}
                )
                raise ValueError(
                    f"Unknown dataset id(s): {sorted(missing)}. "
                    f"Known ids: {known}"
                )
            if not datasets:
                raise ValueError("No datasets matched the requested ids.")
        return [
            {
                "input_path": dataset["input_path"],
                "sequence_col": dataset["sequence_col"],
                "output_path": dataset["output_path"],
                "split_into_subtiles": dataset.get("split_into_subtiles", False),
                "subtile_mode": dataset.get("subtile_mode"),
                "tycko_name_from_cols": dataset.get("tycko_name_from_cols"),
                "subtile_name_col": dataset.get("subtile_name_col", "Gene entry name"),
                "row_filter": dataset.get("row_filter"),
                "balanced_subsample": dataset.get("balanced_subsample"),
                "attach_reference_sequences": dataset.get(
                    "attach_reference_sequences", False
                ),
                "feature_col": dataset.get("feature_col", "#Feature"),
                "drop_input_sequence_col": dataset.get(
                    "drop_input_sequence_col", False
                ),
            }
            for dataset in datasets
        ]

    if not args.sequence_cols or len(args.sequence_cols) != len(args.inputs):
        raise ValueError("Provide exactly one --sequence-col for each --input.")

    if args.outputs and len(args.outputs) != len(args.inputs):
        raise ValueError("Provide either zero --output values or one --output for each --input.")

    jobs = []
    for index, input_path in enumerate(args.inputs):
        jobs.append(
            {
                "input_path": Path(input_path).expanduser(),
                "sequence_col": args.sequence_cols[index],
                "output_path": (
                    Path(args.outputs[index]).expanduser() if args.outputs else None
                ),
                "split_into_subtiles": False,
            }
        )
    return jobs


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)

    for job in jobs:
        input_path = Path(job["input_path"]).expanduser()
        output_path = job["output_path"]
        name_from_cols = job.get("tycko_name_from_cols")
        resolved_output = annotate_dataset(
            input_path=input_path,
            sequence_col=str(job["sequence_col"]),
            output_path=Path(output_path).expanduser() if output_path else None,
            split_into_subtiles=bool(job.get("split_into_subtiles", False)),
            subtile_mode=job.get("subtile_mode"),
            subtile_name_col=str(job.get("subtile_name_col", "Gene entry name")),
            tycko_name_from_cols=tuple(name_from_cols) if name_from_cols else None,
            row_filter=job.get("row_filter"),
            balanced_subsample=job.get("balanced_subsample"),
            attach_reference_sequences=bool(
                job.get("attach_reference_sequences", False)
            ),
            feature_col=str(job.get("feature_col", "#Feature")),
            drop_input_sequence_col=bool(job.get("drop_input_sequence_col", False)),
        )
        print(
            f"Annotated {input_path} using column "
            f"{job['sequence_col']!r} -> {resolved_output}"
        )


if __name__ == "__main__":
    main()
