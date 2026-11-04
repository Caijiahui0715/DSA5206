"""Part II alternatives, member E: Ridge regression on the shared feature sets.

Run from the repository root, after covid_severity_reproduction/reproduce.py and
extract_new_features.py have produced the features:
    python alternatives_E_ridge/ridge_analysis.py

Design (identical to compare_features.py, so all members are comparable):
  * the same 94 images and the same 50 patient-grouped splits (results/splits.json);
  * paper baseline = plain linear regression on the single "Lung Opacity" logit;
  * Ridge penalty alpha is chosen by 5-fold GroupKFold inside the training patients only,
    scored by MAE; features are z-scored inside the pipeline (scaler fitted on training folds);
  * significance against the paper: Nadeau-Bengio corrected resampled t-test.

Analyses (slides 17-19 in the group deck)
  1. Ridge on five feature sets vs the paper              -> summary.csv, fig_ridge_delta.png
  2. Inner-CV MAE against alpha (training patients only)  -> alpha_curve.csv, fig_alpha_curve.png
     and plain OLS vs Ridge on the same features          -> summary.csv (model = ols)
  3. Robustness of Ridge + BiomedCLIP, one choice changed at a time -> robustness.csv
       feature set also chosen by CV / non-journal images only / different inner folds / no z-scoring

Outputs go to alternatives_E_ridge/results/. Member B's script imports the helpers below and
reads per_split_metrics.csv and test_predictions.csv from here.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent / "covid_severity_reproduction"
sys.path.insert(0, str(REPRO))
import compare_features as base  # noqa: E402  (shared loaders, alpha grid and corrected t-test)

OUT = HERE / "results"
TARGETS = base.TARGETS  # (name, CSV column, upper bound of the score)
INNER_FOLDS = 5
PAPER = "paper"
FEATURES = {            # key here -> key in compare_features.load_feature_sets
    "logit1": base.BASELINE,
    "logits18": "densenet18_ridge",
    "dense1024": "densenet1024_ridge",
    "raddino": "raddino_ridge",
    "biomedclip": "biomedclip_ridge",
}
FEATURE_LABELS = {"logit1": "1 logit", "logits18": "18 logits", "dense1024": "1024-dim",
                  "raddino": "RAD-DINO", "biomedclip": "BiomedCLIP"}
COLOURS = {"opacity": "#1f5f99", "extent": "#e08a2c"}


# --------------------------------------------------------------------------- shared helpers
def load_data():
    """Cohort, the 50 splits, patient ids and the five feature matrices (same order as cohort.csv)."""
    cohort = pd.read_csv(base.RESULTS / "cohort.csv")
    splits = json.loads((base.RESULTS / "splits.json").read_text(encoding="utf-8"))
    loaded = base.load_feature_sets(cohort)
    missing = [k for k, v in FEATURES.items() if v not in loaded]
    if missing:
        raise SystemExit(f"Missing feature sets {missing}: run extract_new_features.py first")
    assert len(cohort) == 94 and len(splits) == 50
    return cohort, splits, cohort["patientid"].to_numpy(), {k: loaded[v] for k, v in FEATURES.items()}


def grouped_folds(train, groups, shuffle_seed=None):
    """Inner folds over the training patients. Default: GroupKFold(5), as in compare_features.py.
    With shuffle_seed, patients are randomly re-assigned to the 5 folds (robustness check)."""
    from sklearn.model_selection import GroupKFold

    g = groups[train]
    if shuffle_seed is None:
        return list(GroupKFold(n_splits=INNER_FOLDS).split(train, groups=g))
    patients = np.random.default_rng(shuffle_seed).permutation(np.unique(g))
    fold_of = {p: i % INNER_FOLDS for i, p in enumerate(patients)}
    fold = np.array([fold_of[p] for p in g])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(INNER_FOLDS)]


def tuned(estimator, grid, folds, scale=True, n_jobs=None):
    """GridSearchCV on MAE over grouped inner folds; optional z-scoring inside the pipeline."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("scale", StandardScaler())] if scale else []
    return GridSearchCV(Pipeline(steps + [("model", estimator)]), grid, cv=folds,
                        scoring="neg_mean_absolute_error", n_jobs=n_jobs)


def ridge_search(folds, scale=True):
    from sklearn.linear_model import Ridge

    return tuned(Ridge(), {"model__alpha": base.ALPHAS}, folds, scale)


def fit_quietly(search, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(x, y)
    return search


def metrics(y_true, y_pred, upper):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {"mae": mean_absolute_error(y_true, y_pred), "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "out_of_range_share": float(np.mean((y_pred < 0) | (y_pred > upper)))}


def compare(rows_a, rows_b):
    """Per target: mean of MAE(a) - MAE(b) over splits, splits where a wins, corrected p."""
    out = {}
    for target, _, _ in TARGETS:
        a = rows_a[rows_a.target == target].set_index("seed").sort_index()
        b = rows_b[rows_b.target == target].set_index("seed").sort_index()
        d = (a["mae"] - b["mae"]).to_numpy()
        _, p = base.corrected_t_test(d, a["n_train"].mean(), a["n_test"].mean())
        out[target] = {"mae_minus_ref": float(d.mean()), "splits_better": int((d < 0).sum()),
                       "corrected_p": p}
    return out


def choose_by_inner_cv(rows, label):
    """Per split and target, keep the candidate with the lowest inner-CV MAE (test MAE is not used)."""
    best = rows.loc[rows.groupby(["target", "seed"])["inner_cv_mae"].idxmin()].copy()
    best["chosen"] = best["model"] + " + " + best["features"]
    best["model"], best["features"] = label, "chosen_by_cv"
    return best


class Recorder:
    """Collects per-split metrics and pooled test predictions."""

    def __init__(self, cohort):
        self.cohort, self.rows, self.preds = cohort, [], []

    def add(self, target, upper, seed, y, train, test, model, feature, predicted,
            variant="main", search=None):
        self.rows.append({
            "target": target, "seed": seed, "model": model, "features": feature, "variant": variant,
            "n_train": len(train), "n_test": len(test),
            "inner_cv_mae": -search.best_score_ if search is not None else np.nan,
            "params": json.dumps(search.best_params_ if search is not None else {}, default=float),
            **metrics(y[test], predicted, upper)})
        if variant == "main":
            self.preds.extend({"target": target, "seed": seed, "model": model, "features": feature,
                               "filename": self.cohort["filename"].iloc[i], "y_true": float(y[i]),
                               "y_pred": float(p)} for i, p in zip(test, predicted))

    def frames(self):
        return pd.DataFrame(self.rows), pd.DataFrame(self.preds)


def summary_table(pieces, paper_rows):
    """pieces: list of (model, features, variant, rows). One line per target with the paper comparison."""
    out = []
    for model, feature, variant, part in pieces:
        test = {} if model == PAPER else compare(part, paper_rows)
        for target, _, _ in TARGETS:
            t = part[part.target == target]
            out.append({"target": target, "model": model, "features": feature, "variant": variant,
                        "mae_mean": t["mae"].mean(), "mae_std": t["mae"].std(ddof=1),
                        "r2_mean": t["r2"].mean(), "out_of_range_share": t["out_of_range_share"].mean(),
                        "mae_minus_paper": test.get(target, {}).get("mae_minus_ref", 0.0),
                        "splits_better_than_paper": test.get(target, {}).get("splits_better", 0),
                        "corrected_p_vs_paper": test.get(target, {}).get("corrected_p", np.nan)})
    return pd.DataFrame(out)


def pooled_mae_by_source(preds, cohort):
    """Pooled test MAE split by journal figure (image has a DOI) vs other images."""
    journal = cohort.set_index("filename")["doi"].notna()
    p = preds.assign(journal_figure=preds["filename"].map(journal),
                     abs_error=(preds["y_pred"] - preds["y_true"]).abs())
    return p.groupby(["target", "journal_figure", "model", "features"])["abs_error"].mean().rename("pooled_mae")


def fmt(summary, model, feature, variant="main"):
    s = summary[(summary.model == model) & (summary.features == feature) & (summary.variant == variant)]
    s = s.set_index("target")
    o, e = s.loc["opacity"], s.loc["extent"]
    return (f"MAE {o.mae_mean:.3f} / {e.mae_mean:.3f}   wins {o.splits_better_than_paper} / "
            f"{e.splits_better_than_paper}   p {o.corrected_p_vs_paper:.2f} / {e.corrected_p_vs_paper:.2f}   "
            f"out-of-range {100 * o.out_of_range_share:.1f}% / {100 * e.out_of_range_share:.1f}%")


# --------------------------------------------------------------------------- E analysis
def run():
    from sklearn.linear_model import LinearRegression

    OUT.mkdir(parents=True, exist_ok=True)
    cohort, splits, groups, sets = load_data()
    rec, curve = Recorder(cohort), []
    for split in splits:
        seed = split["seed"]
        train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
        assert set(groups[train]).isdisjoint(groups[test]), "Patient leakage!"
        folds = grouped_folds(train, groups)
        other_folds = grouped_folds(train, groups, shuffle_seed=1000 + seed)
        for target, column, upper in TARGETS:
            y = cohort[column].to_numpy(dtype=float)
            add = lambda *a, **k: rec.add(target, upper, seed, y, train, test, *a, **k)  # noqa: E731

            x = sets["logit1"]
            add(PAPER, "logit1", LinearRegression().fit(x[train], y[train]).predict(x[test]))
            for feature, x in sets.items():
                # plain OLS (minimum-norm least squares when features > images), for OLS vs Ridge
                add("ols", feature, LinearRegression().fit(x[train], y[train]).predict(x[test]))
                search = fit_quietly(ridge_search(folds), x[train], y[train])
                add("ridge", feature, search.predict(x[test]), search=search)
                for a, s in zip(search.cv_results_["param_model__alpha"], search.cv_results_["mean_test_score"]):
                    curve.append({"target": target, "seed": seed, "features": feature,
                                  "alpha": float(a), "inner_cv_mae": -float(s)})

            x = sets["biomedclip"]  # robustness: change one analysis choice at a time
            for variant, search in [("other_inner_folds", ridge_search(other_folds)),
                                    ("no_zscore", ridge_search(folds, scale=False))]:
                fit_quietly(search, x[train], y[train])
                add("ridge", "biomedclip", search.predict(x[test]), variant=variant, search=search)
        if (seed + 1) % 10 == 0:
            print(f"Splits evaluated: {seed + 1}/50", flush=True)

    frame, preds = rec.frames()
    frame.to_csv(OUT / "per_split_metrics.csv", index=False)
    preds.to_csv(OUT / "test_predictions.csv", index=False)

    # Check: paper baseline and Ridge rows must equal A's compare_features.py results
    ref = pd.read_csv(base.OUT / "per_split_metrics.csv")
    for theirs_key, (model, feature) in {base.BASELINE: (PAPER, "logit1"),
                                         "densenet1024_ridge": ("ridge", "dense1024"),
                                         "biomedclip_ridge": ("ridge", "biomedclip")}.items():
        theirs = ref[ref.method == theirs_key].set_index(["target", "seed"])["mae"].sort_index()
        ours = frame[(frame.model == model) & (frame.features == feature) & (frame.variant == "main")]
        np.testing.assert_allclose(ours.set_index(["target", "seed"])["mae"].sort_index().to_numpy(),
                                   theirs.to_numpy(), rtol=0, atol=1e-6)
    print("Check passed: paper baseline and Ridge equal results/part2_features (compare_features.py).")

    # 1-2. summary: paper, Ridge and OLS on every feature set, plus the robustness variants
    main = frame[frame.variant == "main"]
    paper = main[main.model == PAPER]
    ridge_cv = choose_by_inner_cv(main[main.model == "ridge"], "ridge_cv_features")
    ridge_cv[["target", "seed", "chosen", "inner_cv_mae", "mae"]].to_csv(
        OUT / "cv_feature_choice_per_split.csv", index=False)
    pick = lambda m, f, v="main": frame[(frame.model == m) & (frame.features == f) & (frame.variant == v)]  # noqa: E731
    pieces = [(PAPER, "logit1", "main", paper)]
    pieces += [(m, f, "main", pick(m, f)) for m in ["ridge", "ols"] for f in FEATURES]
    pieces += [("ridge", "biomedclip", v, pick("ridge", "biomedclip", v)) for v in ["other_inner_folds", "no_zscore"]]
    pieces += [("ridge_cv_features", "chosen_by_cv", "main", ridge_cv)]
    summary = summary_table(pieces, paper)
    summary.to_csv(OUT / "summary.csv", index=False)

    # alpha: chosen values and the inner-CV curve (mean over 50 training sets)
    ridge_rows = main[main.model == "ridge"].assign(alpha=lambda d: d["params"].map(lambda s: json.loads(s)["model__alpha"]))
    alpha_stats = ridge_rows.groupby(["target", "features"])["alpha"].agg(
        median_alpha="median", share_at_grid_floor=lambda a: float(np.mean(np.isclose(a, base.ALPHAS[0]))))
    alpha_stats.reset_index().to_csv(OUT / "chosen_alpha.csv", index=False)
    curve = pd.DataFrame(curve).groupby(["target", "features", "alpha"])["inner_cv_mae"].mean().reset_index()
    curve.to_csv(OUT / "alpha_curve.csv", index=False)

    # 3. robustness table (slide 19); the non-journal row uses pooled test predictions
    by_source = pooled_mae_by_source(preds, cohort)
    by_source.reset_index().to_csv(OUT / "mae_by_image_source.csv", index=False)
    s = summary.set_index(["model", "features", "variant", "target"])["mae_mean"]
    robust = []
    for label, key in [("BiomedCLIP, as reported", ("ridge", "biomedclip", "main")),
                       ("Feature set also chosen by CV", ("ridge_cv_features", "chosen_by_cv", "main")),
                       ("Different inner folds", ("ridge", "biomedclip", "other_inner_folds")),
                       ("No z-scoring", ("ridge", "biomedclip", "no_zscore"))]:
        robust.append({"variant": label, **{f"paper_{t}": s[(PAPER, "logit1", "main", t)] for t, _, _ in TARGETS},
                       **{f"ridge_{t}": s[key + (t,)] for t, _, _ in TARGETS}})
    robust.insert(2, {"variant": "Non-journal images only (pooled)",
                      **{f"paper_{t}": by_source[(t, False, PAPER, "logit1")] for t, _, _ in TARGETS},
                      **{f"ridge_{t}": by_source[(t, False, "ridge", "biomedclip")] for t, _, _ in TARGETS}})
    robust = pd.DataFrame(robust)
    robust.to_csv(OUT / "robustness.csv", index=False)
    picks = ridge_cv.groupby(["target", "chosen"]).size().rename("splits")
    picks.reset_index().to_csv(OUT / "cv_feature_choice_counts.csv", index=False)

    with pd.option_context("display.width", 200):
        print(f"\n{'paper (1 logit, linear)':28s}", fmt(summary, PAPER, "logit1"))
        for m in ["ridge", "ols"]:
            for f in FEATURES:
                print(f"{m.upper() + ' + ' + FEATURE_LABELS[f]:28s}", fmt(summary, m, f))
        print(f"{'Ridge, features chosen by CV':28s}", fmt(summary, "ridge_cv_features", "chosen_by_cv"))
        print("\nRobustness (opacity / extent):\n" + robust.round(3).to_string(index=False))
        print("\nFeature set chosen by inner CV (splits):\n" + picks.to_string())
        print("\nChosen alpha:\n" + alpha_stats.round(4).to_string())
        for t, _, _ in TARGETS:
            c = curve[(curve.target == t) & (curve.features == "biomedclip")]
            b = c.loc[c.inner_cv_mae.idxmin()]
            print(f"BiomedCLIP inner-CV {t}: {c.inner_cv_mae.iloc[0]:.3f} at the smallest alpha, "
                  f"best {b.inner_cv_mae:.3f} at alpha = {b.alpha:.3g}")
    plots(summary, curve)


def plots(summary, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = summary[summary.variant == "main"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    width, xs = 0.38, np.arange(len(FEATURES))
    for j, (target, _, _) in enumerate(TARGETS):
        vals = [main[(main.model == "ridge") & (main.features == f) & (main.target == target)].mae_minus_paper.item()
                for f in FEATURES]
        bars = ax.bar(xs + (j - 0.5) * width, vals, width, color=COLOURS[target], label=target.capitalize())
        ax.bar_label(bars, fmt="%+.2f", fontsize=7, padding=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs, [FEATURE_LABELS[f] for f in FEATURES])
    ax.set(ylabel="MAE change vs paper (points)",
           title="Ridge: held-out MAE change vs the paper's 1-logit model (below zero is better)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "fig_ridge_delta.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharex=True)
    for ax, (target, _, _) in zip(axes, TARGETS):
        for f in FEATURES:
            c = curve[(curve.target == target) & (curve.features == f)]
            ax.plot(c.alpha, c.inner_cv_mae, label=FEATURE_LABELS[f], linewidth=2.4 if f == "biomedclip" else 1.2)
        ax.set(xscale="log", xlabel="Ridge penalty alpha", ylabel="Inner-CV MAE",
               title=f"{target.capitalize()}: mean of 50 training sets (no test patients)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "fig_alpha_curve.png", dpi=200)
    plt.close(fig)
    print(f"Saved results and figures to {OUT}")


if __name__ == "__main__":
    run()
