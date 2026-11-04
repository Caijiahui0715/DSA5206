# Part II alternatives — E: Ridge regression

Code for the Ridge slides (deck slides 17–19).

```bash
python alternatives_E_ridge/ridge_analysis.py   # run from the repo root, ~10 min on CPU
```

Needs the features from `covid_severity_reproduction/` (`reproduce.py`, `extract_new_features.py`).
Uses the same 50 patient-grouped splits as Part I. Ridge alpha is tuned by 5-fold GroupKFold
inside the training patients only, and the test against the paper is the Nadeau–Bengio corrected t-test.
The script checks that its paper baseline and Ridge rows equal `results/part2_features/` (from `compare_features.py`).

What it computes:
1. Ridge on five feature sets (1 logit, 18 logits, 1024-dim, RAD-DINO, BiomedCLIP) vs the paper's 1-logit linear model
2. Inner-CV MAE against alpha (training patients only), and plain OLS vs Ridge
3. Robustness of Ridge + BiomedCLIP: feature set also chosen by CV, non-journal images only, different inner folds, no z-scoring

Outputs (CSV tables and figures) are written to `alternatives_E_ridge/results/`.
