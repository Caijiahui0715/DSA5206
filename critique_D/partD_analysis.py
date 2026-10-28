"""D 部分（③模型选择 ④模型训练 ⑤模型部署）的证据计算。

输入为 ../covid_severity_reproduction/results 下的 Part I 输出，
沿用同一套 50 次按患者划分（GroupShuffleSplit, seed 0-49）。
只需 numpy、pandas、scikit-learn、matplotlib，不需要 torch。
用法: python partD_analysis.py [results 目录]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

OUT = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "covid_severity_reproduction" / "results"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
TARGETS = {"opacity": ("opacity_mean", 6), "extent": ("geographic_mean", 8)}
NAMES = {"lung_opacity": "1 logit", "four_logits": "4 logits", "eighteen_logits": "18 logits",
         "intermediate_features": "1024 features", "constant": "Constant"}

cohort = pd.read_csv(RESULTS / "cohort.csv")
arrays = np.load(RESULTS / "features.npz")
logits, features = arrays["logits"], arrays["features"]
splits = json.loads((RESULTS / "splits.json").read_text(encoding="utf-8"))
pred = pd.read_csv(RESULTS / "test_predictions.csv")
groups = cohort.patientid.to_numpy()
# 与 reproduce.py 相同的列：Lung Opacity, Pneumonia, Infiltration, Consolidation
four = [16, 8, 2, 1]
INPUTS = {"lung_opacity": logits[:, [16]], "four_logits": logits[:, four], "eighteen_logits": logits,
          "intermediate_features": features, "constant": np.zeros((94, 1))}
summary = {}

# ---------- ④ 选择偏差：按测试集挑 vs 按训练 MSE 挑 vs 嵌套 CV 挑 ----------
rows = []
for s in splits:
    train, test = np.array(s["train_indices"]), np.array(s["test_indices"])
    for target, (col, _) in TARGETS.items():
        y = cohort[col].to_numpy(float)
        per = {}
        for m, x in INPUTS.items():
            model = LinearRegression().fit(x[train], y[train])
            per[m] = {"train_mse": mean_squared_error(y[train], model.predict(x[train])),
                      "test_mae": mean_absolute_error(y[test], model.predict(x[test]))}
            # 训练患者内部 5 折分组 CV，不接触测试集
            inner = []
            for a, b in GroupKFold(5).split(x[train], groups=groups[train]):
                fit = LinearRegression().fit(x[train][a], y[train][a])
                inner.append(mean_squared_error(y[train][b], fit.predict(x[train][b])))
            per[m]["inner_cv_mse"] = float(np.mean(inner))
        by_train = min(per, key=lambda m: per[m]["train_mse"])
        by_cv = min(per, key=lambda m: per[m]["inner_cv_mse"])
        by_test = min(per, key=lambda m: per[m]["test_mae"])
        rows.append({"seed": s["seed"], "target": target,
                     "fixed_lung_opacity_mae": per["lung_opacity"]["test_mae"],
                     "train_mse_choice": by_train, "train_mse_choice_mae": per[by_train]["test_mae"],
                     "nested_cv_choice": by_cv, "nested_cv_mae": per[by_cv]["test_mae"],
                     "test_oracle_choice": by_test, "test_oracle_mae": per[by_test]["test_mae"],
                     **{f"train_mse_{m}": per[m]["train_mse"] for m in per}})
sel = pd.DataFrame(rows)
sel.to_csv(OUT / "selection_per_split.csv", index=False)
for target, g in sel.groupby("target"):
    summary[f"selection_{target}"] = {
        "fixed_lung_opacity_mae": round(g.fixed_lung_opacity_mae.mean(), 3),
        "nested_cv_mae": round(g.nested_cv_mae.mean(), 3),
        "nested_cv_choices": g.nested_cv_choice.value_counts().to_dict(),
        "train_mse_choice_mae": round(g.train_mse_choice_mae.mean(), 3),
        "train_mse_choices": g.train_mse_choice.value_counts().to_dict(),
        "test_oracle_mae": round(g.test_oracle_mae.mean(), 3),
        "test_oracle_choices": g.test_oracle_choice.value_counts().to_dict(),
        "train_mse_1024_mean": float(g.train_mse_intermediate_features.mean()),
    }

# ---------- ④ 残差与校准（最终模型：1 logit + 线性回归）----------
p = pred[pred.method == "lung_opacity"].copy()
p["res"] = p.y_pred - p.y_true
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for i, (target, (_, hi)) in enumerate(TARGETS.items()):
    g = p[p.target == target]
    # 残差 vs 真实分：按真实分整数档分组
    bins = np.floor(g.y_true).clip(upper=hi - 1)
    stat = g.groupby(bins).res.agg(["mean", "std", "count"])
    ax = axes[0, i]
    ax.scatter(g.y_true, g.res, s=6, alpha=0.15, color="#4a6fa5")
    ax.errorbar(stat.index + 0.5, stat["mean"], yerr=stat["std"], fmt="o-", color="#c0392b", capsize=3,
                label="mean ± SD per score band")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set(xlabel="Expert score (y_true)", ylabel="Prediction − truth",
           title=f"{target}: residual vs expert score (50 splits pooled)")
    ax.legend(loc="upper right", fontsize=8)
    # 校准：按预测值分 8 个分位箱，比较箱内平均预测与平均真实分
    q = pd.qcut(g.y_pred, 8)
    cal = g.groupby(q, observed=True).agg(pred=("y_pred", "mean"), true=("y_true", "mean"),
                                           sd=("y_true", "std"))
    slopes = [np.polyfit(gs.y_pred, gs.y_true, 1)[0] for _, gs in g.groupby("seed")]
    ax = axes[1, i]
    ax.plot([-1, hi + 1], [-1, hi + 1], color="grey", ls="--", lw=1, label="perfect calibration")
    ax.errorbar(cal.pred, cal.true, yerr=cal.sd, fmt="o-", color="#27ae60", capsize=3,
                label="binned (octiles of prediction)")
    ax.axvspan(-1, 0, color="#f5b7b1", alpha=0.4)
    ax.axvspan(hi, hi + 1, color="#f5b7b1", alpha=0.4, label="outside valid score range")
    ax.set(xlim=(-1, hi + 1), ylim=(-0.5, hi + 0.5), xlabel="Model prediction", ylabel="Mean expert score",
           title=f"{target}: calibration (slope {np.mean(slopes):.2f} ± {np.std(slopes, ddof=1):.2f})")
    ax.legend(loc="upper left", fontsize=8)
    oor = (g.y_pred < 0) | (g.y_pred > hi)
    summary[f"residual_{target}"] = {
        "mean_resid_y_le_1": round(g.res[g.y_true <= 1].mean(), 3),
        "mean_resid_y_gt_4": round(g.res[g.y_true > 4].mean(), 3),
        "mean_resid_by_band": stat["mean"].round(3).to_dict(),
        "count_by_band": stat["count"].to_dict(),
        "calibration_slope_mean": round(float(np.mean(slopes)), 3),
        "calibration_slope_sd": round(float(np.std(slopes, ddof=1)), 3),
        "out_of_range_share": round(float(oor.mean()), 4),
        "below_0_share": round(float((g.y_pred < 0).mean()), 4),
        "above_max_share": round(float((g.y_pred > hi).mean()), 4),
        "images_ever_out_of_range": int(g[oor].filename.nunique()),
        "mae_raw": round(mean_absolute_error(g.y_true, g.y_pred), 3),
        "mae_clipped": round(mean_absolute_error(g.y_true, g.y_pred.clip(0, hi)), 3),
    }
fig.tight_layout()
fig.savefig(FIG / "residual_calibration.png", dpi=160)

# ---------- ④ 方差：训练误差 vs 测试误差 ----------
mets = pd.read_csv(RESULTS / "metrics_per_split.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, target in zip(axes, TARGETS):
    g = sel[sel.target == target]
    t = mets[mets.target == target].groupby("method").mse.mean()
    ms = list(NAMES)
    xs = np.arange(len(ms))
    ax.bar(xs - 0.2, [g[f"train_mse_{m}"].mean() for m in ms], 0.4, label="train MSE", color="#95a5a6")
    ax.bar(xs + 0.2, [t[m] for m in ms], 0.4, label="test MSE", color="#2c3e50")
    ax.set_xticks(xs, [NAMES[m] for m in ms], rotation=15)
    ax.set(title=f"{target}: train vs test MSE (mean of 50 splits)", ylabel="MSE")
    ax.legend()
fig.tight_layout()
fig.savefig(FIG / "train_vs_test_mse.png", dpi=160)

# ---------- ④ 不平衡与分组表现（1 logit 模型，50 次划分合并）----------
for target, (col, hi) in TARGETS.items():
    band = np.floor(cohort[col]).clip(upper=hi - 1)
    summary[f"label_bands_{target}"] = band.value_counts().sort_index().to_dict()
groups_df = p.merge(cohort[["filename", "sex", "age"]], on="filename")
groups_df["abs_error"] = groups_df.res.abs()
groups_df["sex_group"] = groups_df.sex.fillna("missing")
groups_df["age_group"] = pd.cut(groups_df.age, [0, 50, 65, 120], right=False,
                                labels=["<50", "50-64", "65+"]).astype(str).replace("nan", "missing")
for g in ["sex_group", "age_group"]:
    table = groups_df.groupby([g, "target"]).abs_error.mean().unstack().round(3)
    table["images"] = groups_df.groupby(g).filename.nunique()
    summary[f"mae_by_{g}"] = table.to_dict("index")

(OUT / "partD_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                        encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
