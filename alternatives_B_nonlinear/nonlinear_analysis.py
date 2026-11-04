"""Part II alternatives, member B: nonlinear models on the shared feature sets.

Run from the repository root, after alternatives_E_ridge/ridge_analysis.py (this script reuses
E's helpers and reads E's Ridge and paper-baseline results for the comparisons):
    python alternatives_B_nonlinear/nonlinear_analysis.py

Models: RBF support vector regression (SVR), random forest, gradient boosting.
Features: DenseNet 1 logit, 18 logits, 1024-dim, RAD-DINO 768-dim, BiomedCLIP 512-dim.
Same 50 patient-grouped splits; every hyper-parameter is tuned by 5-fold GroupKFold inside the
training patients only (MAE); corrected resampled t-test against the paper.

Analyses (slides 20-22 in the group deck)
  1. Nonlinear models vs the paper; SVR vs Ridge on BiomedCLIP     -> summary.csv, fig_nonlinear_delta.png
  2. All 20 pipelines (4 models incl. Ridge x 5 feature sets);
     model AND feature set chosen by inner CV (honest selection)   -> mae_grid.csv, fig_mae_grid.png
  3. Out-of-range predictions and under-prediction of severe images
     (true score > 4)                                              -> severe_underprediction.csv,
                                                                      fig_severe_underprediction.png
Outputs go to alternatives_B_nonlinear/results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alternatives_E_ridge"))
import ridge_analysis as E  # noqa: E402  (shared data loading, inner folds, metrics and tests)

OUT = HERE / "results"
SEVERE = 4.0
NONLINEAR = ["svr", "boosting", "forest"]
MODELS = ["ridge"] + NONLINEAR
MODEL_LABELS = {"ridge": "Ridge", "svr": "SVR", "boosting": "Boosting", "forest": "Forest"}


def search_for(model, folds):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR

    if model == "svr":
        return E.tuned(SVR(kernel="rbf"), {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.1, 0.01],
                                           "model__epsilon": [0.1, 0.5]}, folds)
    if model == "forest":
        return E.tuned(RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=0),
                       {"model__min_samples_leaf": [1, 3, 5]}, folds, n_jobs=-1)
    if model == "boosting":
        return E.tuned(GradientBoostingRegressor(n_estimators=100, subsample=0.8, max_features="sqrt",
                                                 random_state=0),
                       {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}, folds, n_jobs=-1)
    raise ValueError(model)


def run():
    e_rows_path, e_preds_path = E.OUT / "per_split_metrics.csv", E.OUT / "test_predictions.csv"
    if not e_rows_path.exists():
        raise SystemExit("Run alternatives_E_ridge/ridge_analysis.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, splits, groups, sets = E.load_data()
    rec = E.Recorder(cohort)
    for split in splits:
        seed = split["seed"]
        train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
        assert set(groups[train]).isdisjoint(groups[test]), "Patient leakage!"
        folds = E.grouped_folds(train, groups)  # identical inner folds to E's Ridge
        for target, column, upper in E.TARGETS:
            y = cohort[column].to_numpy(dtype=float)
            for feature, x in sets.items():
                for model in NONLINEAR:
                    search = E.fit_quietly(search_for(model, folds), x[train], y[train])
                    rec.add(target, upper, seed, y, train, test, model, feature,
                            search.predict(x[test]), search=search)
        if (seed + 1) % 5 == 0:
            print(f"Splits evaluated: {seed + 1}/50", flush=True)

    ours, our_preds = rec.frames()
    ours.to_csv(OUT / "per_split_metrics.csv", index=False)
    our_preds.to_csv(OUT / "test_predictions.csv", index=False)

    # bring in the paper baseline and Ridge from E (same splits, same inner folds)
    e_rows = pd.read_csv(e_rows_path)
    e_rows = e_rows[(e_rows.variant == "main") & e_rows.model.isin([E.PAPER, "ridge"])]
    e_preds = pd.read_csv(e_preds_path)
    e_preds = e_preds[e_preds.model.isin([E.PAPER, "ridge"])]
    rows = pd.concat([e_rows, ours], ignore_index=True)
    preds = pd.concat([e_preds, our_preds], ignore_index=True)
    paper = rows[rows.model == E.PAPER]
    pick = lambda m, f: rows[(rows.model == m) & (rows.features == f)]  # noqa: E731

    # 1-2. summary, 20-pipeline grid and honest selection over all 20 pipelines
    chosen = E.choose_by_inner_cv(rows[rows.model.isin(MODELS)], "cv_model_and_features")
    chosen[["target", "seed", "chosen", "inner_cv_mae", "mae"]].to_csv(OUT / "cv_choice_per_split.csv", index=False)
    pieces = [(E.PAPER, "logit1", "main", paper)]
    pieces += [(m, f, "main", pick(m, f)) for m in MODELS for f in E.FEATURES]
    pieces += [("cv_model_and_features", "chosen_by_cv", "main", chosen)]
    summary = E.summary_table(pieces, paper)
    summary.to_csv(OUT / "summary.csv", index=False)
    grid = summary[summary.model.isin(MODELS)].pivot_table(index=["target", "model"], columns="features",
                                                           values="mae_mean")[list(E.FEATURES)]
    grid.to_csv(OUT / "mae_grid.csv")
    svr_vs_ridge = E.compare(pick("svr", "biomedclip"), pick("ridge", "biomedclip"))
    pd.DataFrame(svr_vs_ridge).T.to_csv(OUT / "svr_vs_ridge_biomedclip.csv")
    picks = chosen.groupby(["target", "chosen"]).size().rename("splits")
    picks.reset_index().to_csv(OUT / "cv_choice_counts.csv", index=False)

    # 3. severe images: mean (true - predicted) over pooled test predictions with true score > 4
    severe = preds[preds.y_true > SEVERE].assign(under=lambda d: d.y_true - d.y_pred)
    severe = severe.groupby(["target", "model", "features"]).agg(
        under_prediction=("under", "mean"), predictions=("under", "size")).reset_index()
    n_images = {t: int((cohort[c] > SEVERE).sum()) for t, c, _ in E.TARGETS}
    severe["severe_images"] = severe.target.map(n_images)
    severe.to_csv(OUT / "severe_underprediction.csv", index=False)

    with pd.option_context("display.width", 200):
        print(f"\n{'paper (1 logit, linear)':30s}", E.fmt(summary, E.PAPER, "logit1"))
        for m in MODELS:
            for f in E.FEATURES:
                print(f"{MODEL_LABELS[m] + ' + ' + E.FEATURE_LABELS[f]:30s}", E.fmt(summary, m, f))
        print(f"{'model + features chosen by CV':30s}", E.fmt(summary, "cv_model_and_features", "chosen_by_cv"))
        print("\nSVR vs Ridge on BiomedCLIP:\n" + pd.DataFrame(svr_vs_ridge).T.round(3).to_string())
        print("\nMAE grid:\n" + grid.round(2).to_string())
        print("\nPipeline chosen by inner CV (splits):\n" + picks.to_string())
        print(f"\nSevere images {n_images}; mean under-prediction:")
        print(severe[(severe.features == "biomedclip") | (severe.model == E.PAPER)].round(2).to_string(index=False))
    plots(summary, grid, severe)


def plots(summary, grid, severe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = 0.38

    def paired_bars(ax, items, values_for, fmt):
        xs = np.arange(len(items))
        for j, (target, _, _) in enumerate(E.TARGETS):
            bars = ax.bar(xs + (j - 0.5) * width, [values_for(target, m, f) for m, f, _ in items], width,
                          color=E.COLOURS[target], label=target.capitalize())
            ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)
        ax.set_xticks(xs, [label for _, _, label in items], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, frameon=False)

    s = summary.set_index(["target", "model", "features"])
    items = [(m, f, f"{MODEL_LABELS[m]}\n{E.FEATURE_LABELS[f]}") for f in ["logit1", "biomedclip"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    paired_bars(ax, items, lambda t, m, f: s.loc[(t, m, f), "mae_minus_paper"], "%+.2f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="MAE change vs paper (points)",
           title="Nonlinear models vs Ridge: held-out MAE change vs the paper (below zero is better)")
    fig.savefig(OUT / "fig_nonlinear_delta.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    for ax, (target, _, _) in zip(axes, E.TARGETS):
        g = grid.loc[target].reindex(MODELS).to_numpy()
        im = ax.imshow(g, cmap="Blues_r", aspect="auto")
        for (i, j), v in np.ndenumerate(g):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(E.FEATURES)), [E.FEATURE_LABELS[f] for f in E.FEATURES], fontsize=8)
        ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS], fontsize=8)
        ax.set_title(f"{target.capitalize()} MAE (paper {s.loc[(target, E.PAPER, 'logit1'), 'mae_mean']:.2f})")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT / "fig_mae_grid.png", dpi=200)
    plt.close(fig)

    sv = severe.set_index(["target", "model", "features"])["under_prediction"]
    order = [(E.PAPER, "logit1", "Paper")] + [(m, "biomedclip", MODEL_LABELS[m]) for m in MODELS]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    paired_bars(ax, order, lambda t, m, f: sv[(t, m, f)], "%.2f")
    ax.set(ylabel="Mean under-prediction (points)",
           title=f"Severe images (true score > {SEVERE:g}); all but the paper use BiomedCLIP")
    fig.savefig(OUT / "fig_severe_underprediction.png", dpi=200)
    plt.close(fig)
    print(f"Saved results and figures to {OUT}")


if __name__ == "__main__":
    run()
