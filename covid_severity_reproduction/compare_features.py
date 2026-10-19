"""Part II: compare feature sets on the paper's 50 patient-grouped splits.

Run after reproduce.py (and, for the new models, after extract_new_features.py):
    .\\.venv\\Scripts\\python.exe .\\compare_features.py

Every method uses the same 50 train/test splits saved in results/splits.json.
The paper baseline is reproduced exactly: plain linear regression on the single
"Lung Opacity" logit. Each other feature set gets Ridge regression whose strength
is chosen only inside the training patients (5-fold GroupKFold), so the test
patients are never used for any choice.
Outputs go to results/part2_features/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "part2_features"
TARGETS = [("opacity", "opacity_mean", 6), ("extent", "geographic_mean", 8)]
LUNG_OPACITY = 16  # index of "Lung Opacity" in torchxrayvision's 18 outputs (checked in reproduce.py)
ALPHAS = np.logspace(-6, 6, 49)  # wide grid; edge hits are reported in summary.csv
BASELINE = "paper_1logit_linear"
LABELS = {
    BASELINE: "Paper: 1 logit + linear regression",
    "densenet18_ridge": "DenseNet 18 logits + Ridge",
    "densenet1024_ridge": "DenseNet 1024 features + Ridge",
    "raddino_ridge": "RAD-DINO + Ridge",
    "biomedclip_ridge": "BiomedCLIP + Ridge",
}
SHORT = {
    BASELINE: "Paper\n1 logit, linear",
    "densenet18_ridge": "DenseNet\n18 logits, Ridge",
    "densenet1024_ridge": "DenseNet\n1024-dim, Ridge",
    "raddino_ridge": "RAD-DINO\nRidge",
    "biomedclip_ridge": "BiomedCLIP\nRidge",
}


def load_feature_sets(cohort):
    with np.load(RESULTS / "features.npz", allow_pickle=False) as arrays:
        features, logits = arrays["features"], arrays["logits"]
    assert features.shape == (94, 1024) and logits.shape == (94, 18)
    sets = {BASELINE: logits[:, [LUNG_OPACITY]], "densenet18_ridge": logits, "densenet1024_ridge": features}
    for name in ["raddino", "biomedclip"]:
        path = OUT / f"features_{name}.npz"
        if not path.exists():
            print(f"Skipping {name}: {path.relative_to(ROOT)} not found (run extract_new_features.py)")
            continue
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["filenames"].tolist() != cohort["filename"].tolist():
                raise ValueError(f"{path.name}: image order does not match results/cohort.csv")
            sets[f"{name}_ridge"] = arrays["features"].astype(np.float64)
    return sets


def ridge_with_inner_cv(x_train, y_train, groups_train):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": ALPHAS},
        cv=GroupKFold(n_splits=5),
        scoring="neg_mean_absolute_error",
    )
    search.fit(x_train, y_train, groups=groups_train)
    return search.best_estimator_, float(search.best_params_["ridge__alpha"])


def safe_pearson(predicted, truth):
    from scipy.stats import pearsonr

    if np.ptp(predicted) == 0:
        return float("nan")
    return float(pearsonr(predicted, truth).statistic)


def corrected_t_test(differences, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated random splits."""
    from scipy import stats

    d = np.asarray(differences, dtype=float)
    j = len(d)
    variance = d.var(ddof=1)
    if variance == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1 / j + n_test / n_train) * variance)
    return float(t), float(2 * stats.t.sf(abs(t), df=j - 1))


def main():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(RESULTS / "cohort.csv")
    splits = json.loads((RESULTS / "splits.json").read_text(encoding="utf-8"))
    assert len(cohort) == 94 and len(splits) == 50
    groups = cohort["patientid"].to_numpy()
    sets = load_feature_sets(cohort)
    print("Feature sets:", ", ".join(f"{k} {v.shape[1]}-dim" for k, v in sets.items()), flush=True)

    rows, predictions = [], []
    for split in splits:
        train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
        assert set(groups[train]).isdisjoint(groups[test]), "Patient leakage!"
        for target, column, upper in TARGETS:
            y = cohort[column].to_numpy(dtype=float)
            for method, x in sets.items():
                alpha = None
                if method == BASELINE:
                    model = LinearRegression().fit(x[train], y[train])
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model, alpha = ridge_with_inner_cv(x[train], y[train], groups[train])
                predicted = model.predict(x[test])
                rows.append({
                    "target": target, "method": method, "seed": split["seed"],
                    "n_train": len(train), "n_test": len(test), "alpha": alpha,
                    "mae": mean_absolute_error(y[test], predicted),
                    "mse": mean_squared_error(y[test], predicted),
                    "r2": r2_score(y[test], predicted),
                    "pearson_r": safe_pearson(predicted, y[test]),
                    "mae_clipped": mean_absolute_error(y[test], np.clip(predicted, 0, upper)),
                    "out_of_range_share": float(np.mean((predicted < 0) | (predicted > upper))),
                })
                predictions.extend({"target": target, "method": method, "seed": split["seed"],
                                    "filename": cohort["filename"].iloc[i], "y_true": float(y[i]),
                                    "y_pred": float(p)} for i, p in zip(test, predicted))
        if (split["seed"] + 1) % 10 == 0:
            print(f"Splits evaluated: {split['seed'] + 1}/50", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "per_split_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "test_predictions.csv", index=False)

    # Sanity check: the baseline must match the Part I reproduction exactly.
    part1 = RESULTS / "metrics_per_split.csv"
    if part1.exists():
        reference = pd.read_csv(part1)
        reference = reference.loc[reference.method == "lung_opacity"].set_index(["target", "seed"])["mae"]
        ours = frame.loc[frame.method == BASELINE].set_index(["target", "seed"])["mae"]
        np.testing.assert_allclose(ours.sort_index().to_numpy(), reference.sort_index().to_numpy(), rtol=1e-9)
        print("Baseline check passed: identical to the Part I reproduction.", flush=True)

    summary = []
    for target, _, _ in TARGETS:
        base = frame.loc[(frame.target == target) & (frame.method == BASELINE)].set_index("seed")
        for method in sets:
            part = frame.loc[(frame.target == target) & (frame.method == method)].set_index("seed")
            difference = (part["mae"] - base["mae"]).to_numpy()
            t, p = (float("nan"), float("nan")) if method == BASELINE else corrected_t_test(
                difference, part["n_train"].mean(), part["n_test"].mean())
            summary.append({
                "target": target, "method": method, "label": LABELS[method],
                "dimensions": sets[method].shape[1],
                "mae_mean": part["mae"].mean(), "mae_std": part["mae"].std(ddof=1),
                "r2_mean": part["r2"].mean(), "pearson_r_mean": part["pearson_r"].mean(),
                "mae_clipped_mean": part["mae_clipped"].mean(),
                "out_of_range_share": part["out_of_range_share"].mean(),
                "mae_minus_paper": difference.mean(),
                "splits_better_than_paper": int((difference < 0).sum()),
                "corrected_t": t, "corrected_p": p,
                "median_alpha": float(part["alpha"].median()) if method != BASELINE else None,
                "alpha_at_grid_edge_share": float(part["alpha"].isin([ALPHAS[0], ALPHAS[-1]]).mean())
                if method != BASELINE else None,
            })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)

    # Leakage check: BiomedCLIP was pretrained on PubMed Central figures, and 40 of the 94
    # images are journal figures (they carry a DOI). Compare errors on the two groups.
    pooled = pd.DataFrame(predictions).merge(
        cohort[["filename"]].assign(journal_figure=cohort["doi"].notna()), on="filename")
    pooled["abs_error"] = (pooled["y_pred"] - pooled["y_true"]).abs()
    by_source = pooled.groupby(["target", "journal_figure", "method"])["abs_error"].mean().unstack("method")
    by_source.to_csv(OUT / "mae_by_image_source.csv")
    print("MAE by image source (journal_figure=True: image has a DOI):")
    print(by_source.round(3).to_string())
    plot(frame, summary, list(sets))

    shown = summary.copy()
    for column in ["mae_mean", "mae_std", "r2_mean", "mae_minus_paper", "corrected_p"]:
        shown[column] = shown[column].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    print(shown[["target", "label", "dimensions", "mae_mean", "mae_std", "r2_mean",
                 "mae_minus_paper", "splits_better_than_paper", "corrected_p"]].to_string(index=False))
    print(f"Saved results to {OUT.relative_to(ROOT)}", flush=True)


def plot(frame, summary, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, (target, _, _) in zip(axes, TARGETS):
        values = [frame.loc[(frame.target == target) & (frame.method == m), "mae"] for m in methods]
        ax.boxplot(values, tick_labels=[SHORT[m] for m in methods])
        baseline = summary.loc[(summary.target == target) & (summary.method == BASELINE), "mae_mean"].item()
        ax.axhline(baseline, color="#d35400", linestyle="--", linewidth=1, label="Paper method, mean MAE")
        ax.set(title=f"{target.capitalize()}: held-out MAE over 50 patient splits", ylabel="MAE (score points)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "mae_by_feature_set.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
