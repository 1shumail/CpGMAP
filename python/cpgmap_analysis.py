#!/usr/bin/env python
"""
CpGMAP — modern reproduction & evaluation (Python).

Predicting CpG methylation status from tissue-specific WGBS methylation
segments (NGSmethDB / Roadmap Epigenomics human tissues).

This script:
  1. Loads all tissue BED dumps from data/raw/*.bed.bz2
  2. Builds features and a binary methylation label
  3. Trains/evaluates classifiers in two honest framings:
       Model A  — features include methRatio (LEAKY, for reference only)
       Model B  — non-leaky region/CpG-structure features (the real task)
  4. Compares Logistic Regression, SVM, and a small neural net (MLP)
  5. Saves figures to results/ and a metrics summary to results/metrics.json

Run:  python python/cpgmap_analysis.py
"""
import os, json, glob, bz2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
RNG = 42

def _viz():
    """Lazy-import plotting libs so the core module + tests don't require them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    return plt, sns
COLS = ["chrom", "start", "end", "id", "score", "strand", "methRatio", "cytosineCount"]
METH_THRESHOLD = 0.5            # a segment is "methylated" if methRatio >= this

# Feature framings. Model B must NEVER use methRatio/score (that would leak the label).
FRAMINGS = {
    "A_leaky_with_methRatio": ["methRatio", "score", "cytosineCount"],
    "B_nonleaky_structure":   ["log_length", "cytosineCount", "log_density"],
}
LEAKY_COLS = {"methRatio", "score"}   # columns that directly encode the label

# ---------------------------------------------------------------- load (I/O)
def parse_bed(path):
    """Read one NGSmethDB BED(.bz2) dump into a DataFrame, tagged with its tissue."""
    # skip the '#chrom...' comment header (skiprows=1) and assign our own names;
    # header=None so the first real data row is kept, not eaten as a header.
    df = pd.read_csv(path, sep="\t", header=None, names=COLS, skiprows=1,
                     dtype={"chrom": str})
    df["tissue"] = os.path.basename(path).split("_")[0]
    return df

def add_features(df):
    """Pure feature engineering + binary methylation label (no I/O, no global state)."""
    df = df.dropna(subset=["methRatio", "cytosineCount", "start", "end"]).copy()
    df["region_length"] = (df["end"] - df["start"]).clip(lower=1)
    df["log_length"] = np.log10(df["region_length"])
    df["cpg_density"] = df["cytosineCount"] / df["region_length"]
    df["log_density"] = np.log10(df["cpg_density"].clip(lower=1e-6))
    df["methylated"] = (df["methRatio"] >= METH_THRESHOLD).astype(int)
    return df

def load_all(raw=RAW):
    frames = []
    for path in sorted(glob.glob(os.path.join(raw, "*.bed.bz2"))):
        df = parse_bed(path)
        frames.append(df)
        print(f"  loaded {df['tissue'].iloc[0]:16s} {len(df):>7,} segments")
    return add_features(pd.concat(frames, ignore_index=True))

# ---------------------------------------------------------------- EDA
def eda(data):
    plt, sns = _viz()
    summary = (data.groupby("tissue")
               .agg(n_segments=("methRatio", "size"),
                    mean_methRatio=("methRatio", "mean"),
                    pct_methylated=("methylated", "mean"),
                    median_cytosineCount=("cytosineCount", "median"),
                    median_length=("region_length", "median"))
               .round(3).reset_index())
    summary["pct_methylated"] = (summary["pct_methylated"] * 100).round(1)
    summary.to_csv(os.path.join(RES, "tissue_summary.csv"), index=False)
    print("\nPer-tissue summary:\n", summary.to_string(index=False))

    # methylation-ratio distribution per tissue
    plt.figure(figsize=(9, 5))
    for t, g in data.groupby("tissue"):
        sns.kdeplot(g["methRatio"], label=t, fill=False, linewidth=1.6)
    plt.title("Methylation-ratio distribution by tissue (NGSmethDB segments)")
    plt.xlabel("methylation ratio"); plt.ylabel("density"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "eda_methratio_by_tissue.png"), dpi=130); plt.close()

    # CpG density vs methylation (the biological signal)
    samp = data.sample(min(40000, len(data)), random_state=RNG)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=samp, x="log_density", y="methRatio", hue="methylated",
                    s=6, alpha=0.25, palette={0: "#2A9D8F", 1: "#B23A2E"}, legend=True)
    plt.title("CpG density vs methylation ratio\n(CpG-dense regions trend unmethylated)")
    plt.xlabel("log10 CpG density"); plt.ylabel("methylation ratio")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "eda_density_vs_meth.png"), dpi=130); plt.close()
    return summary

# ---------------------------------------------------------------- modeling
def build_models(rng=RNG):
    """Return the model zoo (fresh, unfitted) used for every framing."""
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "SVM (RBF)":          make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=rng)),
        "NeuralNet (MLP)":    make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=rng)),
    }

def train_eval(model, X_tr, X_te, y_tr, y_te):
    """Fit one model and score it. Deterministic given inputs + model seed.
    Returns {accuracy, roc_auc, pred, proba}."""
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    try:
        proba = model.predict_proba(X_te)[:, 1]
    except AttributeError:
        proba = model.decision_function(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pred": pred,
        "proba": proba,
    }

def run_models(data):
    plt, sns = _viz()
    # subsample for tractable RBF-SVM training, stratified
    df = data.sample(min(15000, len(data)), random_state=RNG)
    y = df["methylated"].values
    out = {"label_balance_pct_methylated": round(float(y.mean()) * 100, 1)}
    print(f"\nLabel balance: {out['label_balance_pct_methylated']}% methylated (n={len(df):,} sampled)")

    models = build_models()
    out["framings"] = {}
    for fname, feats in FRAMINGS.items():
        print(f"\n  Framing {fname}  features={feats}")
        X = df[feats].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
        res = []
        rocs = {}
        for mname, model in models.items():
            r = train_eval(model, X_tr, X_te, y_tr, y_te)
            proba, pred = r["proba"], r["pred"]
            res.append({"model": mname, "accuracy": round(r["accuracy"], 4), "roc_auc": round(r["roc_auc"], 4)})
            print(f"    {mname:24s} acc={r['accuracy']:.3f}  auc={r['roc_auc']:.3f}")
            rocs[mname] = roc_curve(y_te, proba)
            if fname.startswith("B") and mname == "SVM (RBF)":
                # confusion matrix for the headline honest model
                cm = confusion_matrix(y_te, pred)
                plt.figure(figsize=(4.2, 3.6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["unmeth", "meth"], yticklabels=["unmeth", "meth"])
                plt.title("Confusion matrix — Model B (SVM)"); plt.ylabel("true"); plt.xlabel("predicted")
                plt.tight_layout(); plt.savefig(os.path.join(RES, "confusion_modelB_svm.png"), dpi=130); plt.close()
        out["framings"][fname] = {"features": feats, "results": res}
        # ROC overlay per framing
        plt.figure(figsize=(6, 5))
        for mname, (fpr, tpr, _) in rocs.items():
            auc = [r["roc_auc"] for r in res if r["model"] == mname][0]
            plt.plot(fpr, tpr, label=f"{mname} (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=0.8)
        plt.title(f"ROC — framing {fname}"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(RES, f"roc_{fname}.png"), dpi=130); plt.close()
    return out

# ---------------------------------------------------------------- main
def main():
    print("Loading NGSmethDB tissue methylation segments ...")
    data = load_all()
    print(f"\nTotal: {len(data):,} methylation segments across {data['tissue'].nunique()} tissues")
    summary = eda(data)
    metrics = run_models(data)
    metrics["n_segments_total"] = int(len(data))
    metrics["n_tissues"] = int(data["tissue"].nunique())
    metrics["tissues"] = sorted(data["tissue"].unique().tolist())
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved figures + metrics to results/. Done.")

if __name__ == "__main__":
    main()
