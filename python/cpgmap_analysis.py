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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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
sns.set_theme(style="whitegrid")
RNG = 42
COLS = ["chrom", "start", "end", "id", "score", "strand", "methRatio", "cytosineCount"]

# ---------------------------------------------------------------- load
def load_all():
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.bed.bz2"))):
        tissue = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path, sep="\t", comment=None, header=0, names=COLS, skiprows=1,
                         dtype={"chrom": str})
        df["tissue"] = tissue
        frames.append(df)
        print(f"  loaded {tissue:16s} {len(df):>7,} segments")
    data = pd.concat(frames, ignore_index=True)
    # clean + feature engineering
    data = data.dropna(subset=["methRatio", "cytosineCount", "start", "end"])
    data["region_length"] = (data["end"] - data["start"]).clip(lower=1)
    data["log_length"] = np.log10(data["region_length"])
    data["cpg_density"] = data["cytosineCount"] / data["region_length"]
    data["log_density"] = np.log10(data["cpg_density"].clip(lower=1e-6))
    # binary label: methylated (1) vs unmethylated (0) at 0.5
    data["methylated"] = (data["methRatio"] >= 0.5).astype(int)
    return data

# ---------------------------------------------------------------- EDA
def eda(data):
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
def evaluate(name, model, X_tr, X_te, y_tr, y_te, results):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    try:
        proba = model.predict_proba(X_te)[:, 1]
    except AttributeError:
        proba = model.decision_function(X_te)
    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    results.append({"model": name, "accuracy": round(float(acc), 4), "roc_auc": round(float(auc), 4)})
    print(f"    {name:24s} acc={acc:.3f}  auc={auc:.3f}")
    return proba, pred

def run_models(data):
    # subsample for tractable RBF-SVM training, stratified
    df = data.sample(min(15000, len(data)), random_state=RNG)
    y = df["methylated"].values
    out = {"label_balance_pct_methylated": round(float(y.mean()) * 100, 1)}
    print(f"\nLabel balance: {out['label_balance_pct_methylated']}% methylated (n={len(df):,} sampled)")

    framings = {
        "A_leaky_with_methRatio": ["methRatio", "score", "cytosineCount"],
        "B_nonleaky_structure":   ["log_length", "cytosineCount", "log_density"],
    }
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "SVM (RBF)":          make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RNG)),
        "NeuralNet (MLP)":    make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=RNG)),
    }
    out["framings"] = {}
    best_roc = None
    for fname, feats in framings.items():
        print(f"\n  Framing {fname}  features={feats}")
        X = df[feats].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
        res = []
        rocs = {}
        for mname, model in models.items():
            proba, pred = evaluate(mname, model, X_tr, X_te, y_tr, y_te, res)
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
