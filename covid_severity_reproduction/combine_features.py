"""Part II (exploratory): do combinations of feature sets beat the best single one?

Run after compare_features.py:
    .\\.venv\\Scripts\\python.exe .\\combine_features.py

Same 50 patient-grouped splits and the same inner-CV rule as compare_features.py.
Two ways of combining:
  * Feature fusion: blocks are joined side by side, every column is standardised, and each
    block is down-weighted by 1/sqrt(its width), so a 512-dim block cannot drown a 1-dim one.
    Ridge strength is chosen inside the training patients.
  * Prediction averaging: the paper model's prediction and the BiomedCLIP model's prediction
    are averaged (no extra fitting).
Trying many combinations on the same test splits invites picking a lucky winner, so the
script also reports "honest selection": in each split, the inner CV on the training patients
alone picks one candidate, and only that candidate's test error counts.
Outputs go to results/part2_features/ (files start with "fusion_").
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_features as base  # noqa: E402  (shares loading, targets and the t-test)

OUT = base.OUT
ALPHAS = base.ALPHAS  # same grid as compare_features.py, so single-set rows match summary.csv
SINGLE = {
    base.BASELINE: "Paper: 1 logit + linear",
    "densenet18_ridge": "DenseNet 18 logits",
    "densenet1024_ridge": "DenseNet 1024-dim",
    "raddino_ridge": "RAD-DINO",
    "biomedclip_ridge": "BiomedCLIP",
}
FUSIONS = {
    "fuse_1logit_biomedclip": ([base.BASELINE, "biomedclip_ridge"], "1 logit + BiomedCLIP"),
    "fuse_18logits_biomedclip": (["densenet18_ridge", "biomedclip_ridge"], "18 logits + BiomedCLIP"),
    "fuse_raddino_biomedclip": (["raddino_ridge", "biomedclip_ridge"], "RAD-DINO + BiomedCLIP"),
    "fuse_all_four": (["densenet18_ridge", "densenet1024_ridge", "raddino_ridge", "biomedclip_ridge"],
                      "All four sets"),
}
AVERAGE = "avg_paper_biomedclip"
SELECT = "honest_selection"
LABELS = {**SINGLE, **{k: v[1] for k, v in FUSIONS.items()},
          AVERAGE: "Average of paper + BiomedCLIP predictions", SELECT: "Honest selection (inner CV picks)"}


def scale_blocks(x, weights):
    return x * weights


def fit_ridge(x_train, y_train, groups_train, weights=None):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    steps = [StandardScaler()]
    if weights is not None:
        steps.append(FunctionTransformer(scale_blocks, kw_args={"weights": weights}))
    steps.append(Ridge())
    search = GridSearchCV(make_pipeline(*steps), {"ridge__alpha": ALPHAS},
                          cv=GroupKFold(n_splits=5), scoring="neg_mean_absolute_error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(x_train, y_train, groups=groups_train)
    return search.best_estimator_, float(search.best_params_["ridge__alpha"]), float(-search.best_score_)


def fit_linear(x_train, y_train, groups_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold, cross_val_score

    inner = -cross_val_score(LinearRegression(), x_train, y_train, groups=groups_train,
                             cv=GroupKFold(n_splits=5), scoring="neg_mean_absolute_error").mean()
    return LinearRegression().fit(x_train, y_train), None, float(inner)


def main():
    from sklearn.metrics import mean_absolute_error, r2_score

    cohort = pd.read_csv(base.RESULTS / "cohort.csv")
    splits = json.loads((base.RESULTS / "splits.json").read_text(encoding="utf-8"))
    groups = cohort["patientid"].to_numpy()
    sets = base.load_feature_sets(cohort)
    missing = [k for k in SINGLE if k not in sets]
    if missing:
        raise SystemExit(f"Missing feature sets {missing}: run extract_new_features.py first")
    inputs = {k: (sets[k], None) for k in SINGLE}
    for name, (parts, _) in FUSIONS.items():
        blocks = [sets[p] for p in parts]
        weights = np.concatenate([np.full(b.shape[1], 1 / np.sqrt(b.shape[1])) for b in blocks])
        inputs[name] = (np.hstack(blocks), weights)
    print("Candidates:", ", ".join(f"{k} {x.shape[1]}-dim" for k, (x, _) in inputs.items()), flush=True)

    rows, picks = [], []
    for split in splits:
        train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
        assert set(groups[train]).isdisjoint(groups[test]), "Patient leakage!"
        for target, column, upper in base.TARGETS:
            y = cohort[column].to_numpy(dtype=float)
            predicted, inner = {}, {}
            for name, (x, weights) in inputs.items():
                if name == base.BASELINE:
                    model, alpha, inner[name] = fit_linear(x[train], y[train], groups[train])
                else:
                    model, alpha, inner[name] = fit_ridge(x[train], y[train], groups[train], weights)
                predicted[name] = model.predict(x[test])
            predicted[AVERAGE] = (predicted[base.BASELINE] + predicted["biomedclip_ridge"]) / 2
            chosen = min(inner, key=inner.get)
            predicted[SELECT] = predicted[chosen]
            picks.append({"target": target, "seed": split["seed"], "chosen": chosen, "inner_mae": inner[chosen]})
            for name, values in predicted.items():
                rows.append({"target": target, "method": name, "seed": split["seed"],
                             "n_train": len(train), "n_test": len(test),
                             "mae": mean_absolute_error(y[test], values), "r2": r2_score(y[test], values),
                             "inner_cv_mae": inner.get(name)})
        if (split["seed"] + 1) % 10 == 0:
            print(f"Splits evaluated: {split['seed'] + 1}/50", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fusion_per_split.csv", index=False)
    picks = pd.DataFrame(picks)
    counts = picks.groupby(["target", "chosen"]).size().rename("splits").reset_index()
    counts["label"] = counts["chosen"].map(LABELS)
    counts.to_csv(OUT / "fusion_selection_counts.csv", index=False)

    summary = []
    for target, _, _ in base.TARGETS:
        ref = frame.loc[(frame.target == target) & (frame.method == base.BASELINE)].set_index("seed")
        best_single = frame.loc[(frame.target == target) & (frame.method == "biomedclip_ridge")].set_index("seed")
        for name in LABELS:
            part = frame.loc[(frame.target == target) & (frame.method == name)].set_index("seed")
            vs_paper = (part["mae"] - ref["mae"]).to_numpy()
            vs_bclip = (part["mae"] - best_single["mae"]).to_numpy()
            n_train, n_test = part["n_train"].mean(), part["n_test"].mean()
            summary.append({
                "target": target, "method": name, "label": LABELS[name],
                "mae_mean": part["mae"].mean(), "mae_std": part["mae"].std(ddof=1), "r2_mean": part["r2"].mean(),
                "mae_minus_paper": vs_paper.mean(), "splits_better_than_paper": int((vs_paper < 0).sum()),
                "p_vs_paper": np.nan if name == base.BASELINE else base.corrected_t_test(vs_paper, n_train, n_test)[1],
                "mae_minus_biomedclip": vs_bclip.mean(),
                "splits_better_than_biomedclip": int((vs_bclip < 0).sum()),
                "p_vs_biomedclip": np.nan if name == "biomedclip_ridge" else base.corrected_t_test(vs_bclip, n_train, n_test)[1],
            })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "fusion_summary.csv", index=False)
    plot(frame, summary)

    shown = summary.copy()
    for c in ["mae_mean", "mae_std", "r2_mean", "mae_minus_paper", "p_vs_paper", "mae_minus_biomedclip", "p_vs_biomedclip"]:
        shown[c] = shown[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    print(shown[["target", "label", "mae_mean", "mae_std", "r2_mean", "mae_minus_paper", "splits_better_than_paper",
                 "p_vs_paper", "mae_minus_biomedclip", "splits_better_than_biomedclip", "p_vs_biomedclip"]].to_string(index=False))
    print("\nWhat honest selection picked (splits out of 50):")
    print(counts[["target", "label", "splits"]].to_string(index=False))
    print(f"Saved fusion results to {OUT.relative_to(base.ROOT)}", flush=True)


def plot(frame, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, (target, _, _) in zip(axes, base.TARGETS):
        means = [summary.loc[(summary.target == target) & (summary.method == m), "mae_mean"].item() for m in order]
        values = [frame.loc[(frame.target == target) & (frame.method == m), "mae"] for m in order]
        positions = np.arange(len(order))[::-1]
        try:
            ax.boxplot(values, positions=positions, orientation="horizontal", widths=0.6)
        except TypeError:  # matplotlib < 3.10
            ax.boxplot(values, positions=positions, vert=False, widths=0.6)
        ax.set_yticks(positions, [LABELS[m] for m in order], fontsize=8)
        ax.axvline(means[0], color="#d35400", linestyle="--", linewidth=1, label="Paper method, mean MAE")
        ax.set(title=f"{target.capitalize()}: held-out MAE, 50 patient splits", xlabel="MAE (score points)")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "fusion_mae.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
