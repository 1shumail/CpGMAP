"""
Test suite for CpGMAP (python/cpgmap_analysis.py).

Covers each implemented function:
  - parse_bed        : reads an NGSmethDB BED dump into the right schema
  - add_features     : region_length, cpg_density, log features, methylation label
  - FRAMINGS         : the no-data-leakage design guarantee
  - build_models     : the model zoo is well-formed
  - train_eval       : models fit, score, and are reproducible
  - end-to-end       : parse -> features -> split -> train on a tiny synthetic set

These are fast CPU unit tests (no real data, no network) — they check code
correctness, not model accuracy. Run with:  pytest -q
"""
import os
import sys
import bz2
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import cpgmap_analysis as cg  # noqa: E402


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def raw_df():
    """A tiny raw frame in the NGSmethDB BED schema (as parse_bed would return)."""
    return pd.DataFrame({
        "chrom":         ["chr1", "chr1", "chr2", "chr2"],
        "start":         [100,    200,    0,      50],
        "end":           [1100,   250,    100,    60],
        "id":            ["a",    "b",    "c",    "d"],
        "score":         [800,    420,    100,    950],
        "strand":        [".",    ".",    ".",    "."],
        "methRatio":     [0.80,   0.42,   0.10,   0.95],
        "cytosineCount": [20,     5,      2,      30],
        "tissue":        ["spleen"] * 4,
    })


def _synthetic_xy(n=200, seed=0):
    """Two well-separated Gaussian blobs -> an easy 2-class problem."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.0, 1.0, (n, 3)), rng.normal(5.0, 1.0, (n, 3))])
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    return X, y


# ----------------------------------------------------------------- parse_bed
def test_parse_bed_schema_and_tissue(tmp_path):
    content = ("#chrom\tstart\tend\tid\tscore\tstrand\tmethRatio\tcytosineCount\n"
               "chr1\t100\t1100\tchr1_100\t800\t.\t0.8\t20\n"
               "chr1\t200\t250\tchr1_200\t420\t.\t0.42\t5\n")
    p = tmp_path / "spleen_99.bed.bz2"
    with bz2.open(p, "wt") as f:
        f.write(content)
    df = cg.parse_bed(str(p))
    assert list(df.columns) == cg.COLS + ["tissue"]
    assert len(df) == 2
    assert df["tissue"].iloc[0] == "spleen"        # tissue parsed from filename
    assert df["methRatio"].iloc[0] == pytest.approx(0.8)
    assert isinstance(df["chrom"].iloc[0], str)    # chrom kept as string, not numeric


# ----------------------------------------------------------------- add_features
def test_add_features_creates_expected_columns(raw_df):
    out = cg.add_features(raw_df)
    for col in ["region_length", "log_length", "cpg_density", "log_density", "methylated"]:
        assert col in out.columns


def test_region_length_is_end_minus_start(raw_df):
    out = cg.add_features(raw_df)
    assert list(out["region_length"]) == [1000, 50, 100, 10]


def test_region_length_clipped_to_at_least_one():
    df = pd.DataFrame({
        "chrom": ["chr1"], "start": [500], "end": [500], "id": ["x"], "score": [500],
        "strand": ["."], "methRatio": [0.5], "cytosineCount": [3], "tissue": ["spleen"],
    })
    out = cg.add_features(df)
    assert (out["region_length"] >= 1).all()       # no zero/negative lengths -> no div-by-zero


def test_cpg_density_is_cytosines_over_length(raw_df):
    out = cg.add_features(raw_df)
    np.testing.assert_allclose(out["cpg_density"].values,
                               [20 / 1000, 5 / 50, 2 / 100, 30 / 10])


def test_methylation_label_threshold(raw_df):
    out = cg.add_features(raw_df)
    assert list(out["methylated"]) == [1, 0, 0, 1]  # >= 0.5 -> methylated


def test_methylation_label_boundary():
    df = pd.DataFrame({
        "chrom": ["c"] * 3, "start": [0, 0, 0], "end": [10, 10, 10], "id": ["a", "b", "c"],
        "score": [500, 499, 501], "strand": ["."] * 3,
        "methRatio": [0.500, 0.499, 0.501], "cytosineCount": [1, 1, 1], "tissue": ["t"] * 3,
    })
    out = cg.add_features(df)
    assert list(out["methylated"]) == [1, 0, 1]     # exactly 0.5 counts as methylated


def test_add_features_drops_null_methratio():
    df = pd.DataFrame({
        "chrom": ["c", "c"], "start": [0, 0], "end": [10, 10], "id": ["a", "b"],
        "score": [800, np.nan], "strand": ["."] * 2,
        "methRatio": [0.8, np.nan], "cytosineCount": [5, 5], "tissue": ["t", "t"],
    })
    out = cg.add_features(df)
    assert len(out) == 1
    assert out["methRatio"].notna().all()


def test_add_features_single_row(raw_df):
    out = cg.add_features(raw_df.iloc[[0]])
    assert len(out) == 1
    assert out["methylated"].iloc[0] == 1


# ----------------------------------------------------------------- no leakage
def test_model_B_uses_no_leaky_features():
    """The honest model must never see methRatio/score (that IS the label)."""
    feats = set(cg.FRAMINGS["B_nonleaky_structure"])
    assert feats.isdisjoint(cg.LEAKY_COLS), f"leak: {feats & cg.LEAKY_COLS}"


def test_model_A_is_the_leaky_reference():
    assert "methRatio" in cg.FRAMINGS["A_leaky_with_methRatio"]


def test_framing_features_exist_after_engineering(raw_df):
    out = cg.add_features(raw_df)
    for feats in cg.FRAMINGS.values():
        assert all(f in out.columns for f in feats)


# ----------------------------------------------------------------- models
def test_build_models_returns_three_estimators():
    models = cg.build_models()
    assert set(models) == {"LogisticRegression", "SVM (RBF)", "NeuralNet (MLP)"}
    for m in models.values():
        assert hasattr(m, "fit") and hasattr(m, "predict")


def test_train_eval_smoke_and_ranges():
    X, y = _synthetic_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    res = cg.train_eval(cg.build_models()["LogisticRegression"], Xtr, Xte, ytr, yte)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert len(res["pred"]) == len(yte)
    assert res["accuracy"] > 0.9                    # blobs are easily separable


def test_train_eval_is_reproducible():
    X, y = _synthetic_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    a = cg.train_eval(cg.build_models()["NeuralNet (MLP)"], Xtr, Xte, ytr, yte)
    b = cg.train_eval(cg.build_models()["NeuralNet (MLP)"], Xtr, Xte, ytr, yte)
    assert a["accuracy"] == b["accuracy"]           # fixed seed -> identical result


# ----------------------------------------------------------------- end to end
def test_end_to_end_on_synthetic_segments():
    """parse -> features -> split on Model B features -> train -> finite metrics."""
    rng = np.random.default_rng(1)
    n, half = 400, 200
    # methylated group: low CpG density (long regions, few cytosines), high methRatio
    # unmethylated group: high CpG density (short regions, many cytosines), low methRatio
    length = np.r_[rng.integers(2000, 5000, half), rng.integers(200, 800, half)]
    cyto = np.r_[rng.integers(1, 10, half), rng.integers(30, 60, half)]
    methratio = np.clip(np.r_[rng.uniform(0.60, 0.95, half),
                              rng.uniform(0.05, 0.40, half)] + rng.normal(0, 0.05, n), 0, 1)
    df = pd.DataFrame({
        "chrom": ["chr1"] * n, "start": np.zeros(n, int), "end": length,
        "id": [f"r{i}" for i in range(n)], "score": (methratio * 1000).astype(int),
        "strand": ["."] * n, "methRatio": methratio, "cytosineCount": cyto, "tissue": ["spleen"] * n,
    })
    data = cg.add_features(df)
    feats = cg.FRAMINGS["B_nonleaky_structure"]
    X, ylab = data[feats].values, data["methylated"].values
    Xtr, Xte, ytr, yte = train_test_split(X, ylab, test_size=0.3, stratify=ylab, random_state=0)
    res = cg.train_eval(cg.build_models()["LogisticRegression"], Xtr, Xte, ytr, yte)
    assert np.isfinite(res["accuracy"]) and np.isfinite(res["roc_auc"])
    assert res["accuracy"] >= ylab.mean() - 0.05    # at least near the majority baseline
