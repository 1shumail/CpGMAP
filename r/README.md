# R model (reconstructed, for the record)

A clean reconstruction of the original 2018 R analysis: a `neuralnet` classifier
(the surviving model) plus an `e1071` SVM (the proposal's intended method) on the
non-leaky structural features.

```bash
Rscript cpgmap.R   # reads ../data/raw, writes ../results/r_neuralnet.png
```

## Tests
Feature functions live in `features.R` and are unit-tested with `testthat`:
```bash
Rscript -e 'testthat::test_dir("r/tests/testthat")'
```
