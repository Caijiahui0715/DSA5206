"""共享 PPT 模板用的两张 D 部分小图（配色与模板一致）。

用法: python slide_figures.py [results 目录]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "covid_severity_reproduction" / "results"
NAVY, ORANGE, GREY, BG = "#14213D", "#B5521B", "#6B7280", "#FAF9F6"
TARGETS = {"opacity": 6, "extent": 8}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GREY,
                     "axes.labelcolor": NAVY, "xtick.color": GREY, "ytick.color": GREY})

pred = pd.read_csv(RESULTS / "test_predictions.csv")
pred = pred[pred.method == "lung_opacity"].copy()
pred["res"] = pred.y_pred - pred.y_true

# 1. 按专家评分档的平均误差
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), facecolor=BG)
for ax, (target, hi) in zip(axes, TARGETS.items()):
    g = pred[pred.target == target]
    band = np.floor(g.y_true).clip(upper=hi - 1)
    mean = g.groupby(band).res.mean()
    slope = np.mean([np.polyfit(s.y_pred, s.y_true, 1)[0] for _, s in g.groupby("seed")])
    ax.set_facecolor(BG)
    ax.bar(mean.index, mean.values, color=[NAVY if v > 0 else ORANGE for v in mean.values], width=0.7)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(mean.index, [f"{int(b)}" for b in mean.index])
    ax.set(xlabel=f"Expert {target} score band", ylim=(-1.7, 1.2))
    ax.set_title(f"{target}: calibration slope {slope:.2f}", color=NAVY, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Mean prediction − truth")
fig.tight_layout()
fig.savefig(OUT / "figures" / "slide_residual_by_band.png", dpi=220, facecolor=BG)

# 2. 预测值分布与合法范围
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), facecolor=BG)
for ax, (target, hi) in zip(axes, TARGETS.items()):
    g = pred[pred.target == target]
    out = ((g.y_pred < 0) | (g.y_pred > hi)).mean()
    ax.set_facecolor(BG)
    ax.axvspan(g.y_pred.min() - 0.5, 0, color=ORANGE, alpha=0.18, lw=0)
    ax.axvspan(hi, g.y_pred.max() + 0.5, color=ORANGE, alpha=0.18, lw=0)
    ax.hist(g.y_pred, bins=np.arange(np.floor(g.y_pred.min()), np.ceil(g.y_pred.max()) + 0.25, 0.25), color=NAVY)
    ax.set(xlabel=f"Predicted {target} score (valid 0–{hi})", xlim=(g.y_pred.min() - 0.5, g.y_pred.max() + 0.5))
    ax.set_title(f"{out:.1%} of predictions out of range", color=ORANGE, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Test predictions")
fig.tight_layout()
fig.savefig(OUT / "figures" / "slide_prediction_range.png", dpi=220, facecolor=BG)
print("ok")
