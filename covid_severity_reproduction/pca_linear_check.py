"""Part II: do the new features help when the model stays exactly the paper's (plain linear regression)?

Run after compare_features.py:
    .\\.venv\\Scripts\\python.exe .\\pca_linear_check.py

Plain least squares cannot take 512 or 768 columns with ~47 training images, so each new
feature set is first reduced to its top k principal components. PCA is fitted on the training
images of each split only. Models, all ordinary LinearRegression like the paper:
  * paper:            the single "Lung Opacity" logit (k = 0)
  * logit + k PCs:    the logit plus the top k components of RAD-DINO or BiomedCLIP
  * k PCs only:       the components alone
  * logit + PCs, k chosen inside the training patients (5-fold GroupKFold): the honest version
Same 50 patient-grouped splits as Part I. Outputs: results/part2_features/pca_linear_*.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_features as base  # noqa: E402

KS = [1, 2, 3, 5, 10, 20]


def design(logit_train, logit_test, emb_train, emb_test, k, with_logit):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, random_state=0).fit(emb_train)
    parts_train, parts_test = [pca.transform(emb_train)], [pca.transform(emb_test)]
    if with_logit:
        parts_train.insert(0, logit_train)
        parts_test.insert(0, logit_test)
    return np.hstack(parts_train), np.hstack(parts_test)


def fit_predict(logit, emb, y, train, test, k, with_logit):
    from sklearn.linear_model import LinearRegression

    x_train, x_test = design(logit[train], logit[test], emb[train], emb[test], k, with_logit)
    return LinearRegression().fit(x_train, y[train]).predict(x_test)


def inner_choose_k(logit, emb, y, train, groups):
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold

    scores = {}
    folds = list(GroupKFold(n_splits=5).split(train, groups=groups[train]))
    for k in [0] + KS:
        errors = []
        for inner_train, inner_val in folds:
            tr, va = train[inner_train], train[inner_val]
            if k == 0:
                from sklearn.linear_model import LinearRegression
                pred = LinearRegression().fit(logit[tr], y[tr]).predict(logit[va])
            else:
                pred = fit_predict(logit, emb, y, tr, va, k, True)
            errors.append(mean_absolute_error(y[va], pred))
        scores[k] = float(np.mean(errors))
    return min(scores, key=scores.get)


def main():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score

    cohort = pd.read_csv(base.RESULTS / "cohort.csv")
    splits = json.loads((base.RESULTS / "splits.json").read_text(encoding="utf-8"))
    groups = cohort["patientid"].to_numpy()
    sets = base.load_feature_sets(cohort)
    logit = sets[base.BASELINE]
    embeddings = {name: sets[f"{name}_ridge"] for name in ["biomedclip", "raddino"] if f"{name}_ridge" in sets}
    if not embeddings:
        raise SystemExit("Run extract_new_features.py first")

    rows, chosen = [], []
    for split in splits:
        train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
        for target, column, _ in base.TARGETS:
            y = cohort[column].to_numpy(dtype=float)
            preds = {"paper (1 logit)": LinearRegression().fit(logit[train], y[train]).predict(logit[test])}
            for name, emb in embeddings.items():
                for k in KS:
                    preds[f"{name}: logit + {k} PCs"] = fit_predict(logit, emb, y, train, test, k, True)
                    preds[f"{name}: {k} PCs only"] = fit_predict(logit, emb, y, train, test, k, False)
                k_star = inner_choose_k(logit, emb, y, train, groups)
                chosen.append({"target": target, "seed": split["seed"], "features": name, "k": k_star})
                preds[f"{name}: logit + PCs, k chosen in training"] = (
                    preds["paper (1 logit)"] if k_star == 0 else preds[f"{name}: logit + {k_star} PCs"])
            for method, p in preds.items():
                rows.append({"target": target, "method": method, "seed": split["seed"],
                             "n_train": len(train), "n_test": len(test),
                             "mae": mean_absolute_error(y[test], p), "r2": r2_score(y[test], p)})
        if (split["seed"] + 1) % 10 == 0:
            print(f"Splits evaluated: {split['seed'] + 1}/50", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(base.OUT / "pca_linear_per_split.csv", index=False)
    chosen = pd.DataFrame(chosen)
    chosen.groupby(["target", "features", "k"]).size().rename("splits").reset_index().to_csv(
        base.OUT / "pca_linear_chosen_k.csv", index=False)

    summary = []
    for target, _, _ in base.TARGETS:
        ref = frame.loc[(frame.target == target) & (frame.method == "paper (1 logit)")].set_index("seed")["mae"]
        for method in frame.method.unique():
            part = frame.loc[(frame.target == target) & (frame.method == method)].set_index("seed")
            d = (part["mae"] - ref).to_numpy()
            p = np.nan if method == "paper (1 logit)" else base.corrected_t_test(d, part["n_train"].mean(), part["n_test"].mean())[1]
            summary.append({"target": target, "method": method, "mae_mean": part["mae"].mean(),
                            "mae_std": part["mae"].std(ddof=1), "r2_mean": part["r2"].mean(),
                            "mae_minus_paper": d.mean(), "splits_better_than_paper": int((d < 0).sum()),
                            "p_vs_paper": p})
    summary = pd.DataFrame(summary)
    summary.to_csv(base.OUT / "pca_linear_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string(index=False))
        print("\nk chosen inside the training patients (splits out of 50):")
        print(chosen.groupby(["target", "features"])["k"].value_counts().sort_index().to_string())
    print(f"Saved to {base.OUT.relative_to(base.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
