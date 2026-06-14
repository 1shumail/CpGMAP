# Python analysis (current)

Reproducible methylation-status analysis on the 7-tissue NGSmethDB data.

```bash
conda env create -f ../environment.yml && conda activate cpgmap
python cpgmap_analysis.py
```

It performs EDA, builds features, and compares Logistic Regression / SVM / MLP under
two framings (a leaky reference and the honest non-leaky task). Figures and a
`metrics.json` are written to `../results/`. See the root [README](../README.md).
