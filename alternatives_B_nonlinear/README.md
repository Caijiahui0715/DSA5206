# Part II alternatives — B: nonlinear models

Code for the nonlinear-model slides (deck slides 20–22).

```bash
python alternatives_E_ridge/ridge_analysis.py           # run E first (paper baseline + Ridge)
python alternatives_B_nonlinear/nonlinear_analysis.py   # ~45 min on a 2-core CPU
```

Models: RBF SVR, random forest, and gradient boosting, on the same five feature sets and 50 patient splits.
All hyper-parameters are tuned by 5-fold GroupKFold inside the training patients. The script reuses E's helpers and reads E's Ridge results.

What it computes:
1. Nonlinear models vs the paper, and SVR vs Ridge on BiomedCLIP (corrected t-test)
2. All 20 pipelines (4 models × 5 feature sets), plus model and feature set chosen by inner CV
3. Out-of-range predictions and under-prediction of severe images (true score > 4)

Outputs (CSV tables and figures) are written to `alternatives_B_nonlinear/results/`.
