"""C 部分（① 数据收集 ② 数据准备）的证据计算。

输入为 ../covid_severity_reproduction 下 Part I 使用的 94 张评分图、历史元数据和测试集预测。
只需 numpy、pandas、matplotlib、pillow。
用法: python partC_analysis.py [covid_severity_reproduction 目录]
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
REPRO = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "covid_severity_reproduction"
RESULTS, IMAGES = REPRO / "results", REPRO / "data" / "images"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

cohort = pd.read_csv(RESULTS / "cohort.csv")
assert len(cohort) == 94
summary = {}

# ---------- ① 数据来源 ----------
JOURNALS = {"www.nejm.org", "www.thelancet.com", "pubs.rsna.org", "www.sciencedirect.com", "academic.oup.com",
            "onlinelibrary.wiley.com", "www.nature.com", "www.kjronline.org"}


def source(url):
    host = urlparse(url).netloc
    if host in JOURNALS:
        return "Journal article figures"
    return {"www.sirm.org": "SIRM case reports (Italy)", "radiopaedia.org": "Radiopaedia",
            "www.ams.edu.sg": "Academy of Medicine Singapore PDF"}.get(host, "Other case-sharing sites")


def country(location):
    if pd.isna(location):
        return "Not recorded"
    tail = location.split(",")[-1].strip()
    return {"USA": "USA", "Macao": "China"}.get(tail, tail)


cohort["host"] = cohort.url.map(lambda u: urlparse(u).netloc)
cohort["source"] = cohort.url.map(source)
cohort["country"] = cohort.location.map(country)
summary["images_by_source"] = cohort.source.value_counts().to_dict()
summary["images_by_host"] = cohort.host.value_counts().to_dict()
summary["patients_by_source"] = cohort.groupby("source").patientid.nunique().to_dict()
summary["images_by_country"] = cohort.country.value_counts().to_dict()
summary["license_field"] = cohort.license.fillna("Not recorded").value_counts().to_dict()
summary["images_per_patient"] = cohort.groupby("patientid").size().value_counts().sort_index().to_dict()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, title in [(axes[0], "source", "Images by source"), (axes[1], "country", "Images by country")]:
    counts = cohort[col].value_counts().sort_values()
    colors = ["#B5521B" if k == "Not recorded" else "#14213D" for k in counts.index]
    ax.barh(counts.index, counts.values, color=colors)
    for y, v in enumerate(counts.values):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=9)
    ax.set(title=f"{title} (n = 94)", xlabel="Images")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "data_sources.png", dpi=160)

# ---------- ① 人口学：论文 vs 实际 ----------
patients = cohort.groupby("patientid").agg(sex=("sex", "first"), age=("age", "first"))
rows = [
    {"item": "Male / Female", "paper": "44 / 36",
     "images": f"{(cohort.sex == 'M').sum()} / {(cohort.sex == 'F').sum()}",
     "patients": f"{(patients.sex == 'M').sum()} / {(patients.sex == 'F').sum()}"},
    {"item": "Sex missing", "paper": "0 (\"no missing data\")",
     "images": int(cohort.sex.isna().sum()), "patients": int(patients.sex.isna().sum())},
    {"item": "Age, mean ± SD", "paper": "56 ± 14.8",
     "images": f"{cohort.age.mean():.1f} ± {cohort.age.std():.1f}",
     "patients": f"{patients.age.mean():.1f} ± {patients.age.std():.1f}"},
    {"item": "Age missing", "paper": "0", "images": int(cohort.age.isna().sum()),
     "patients": int(patients.age.isna().sum())},
]
for sex, paper in [("M", "55 ± 15.6"), ("F", "57 ± 13.9")]:
    rows.append({"item": f"Age ({sex}), mean ± SD", "paper": paper,
                 "images": f"{cohort.age[cohort.sex == sex].mean():.1f} ± {cohort.age[cohort.sex == sex].std():.1f}",
                 "patients": f"{patients.age[patients.sex == sex].mean():.1f} ± {patients.age[patients.sex == sex].std():.1f}"})
demographics = pd.DataFrame(rows)
demographics.to_csv(OUT / "demographics_check.csv", index=False)
summary["demographics"] = demographics.to_dict("records")
summary["missing_fields_images"] = cohort[["sex", "age", "location", "survival", "went_icu", "intubated",
                                           "offset", "license"]].isna().sum().to_dict()

# ---------- ② 图像质量 ----------
props = []
for f in cohort.filename:
    with Image.open(IMAGES / f) as im:
        props.append({"filename": f, "format": im.format, "mode": im.mode, "width": im.size[0],
                      "height": im.size[1], "short_side": min(im.size)})
props = pd.DataFrame(props).merge(cohort[["filename", "source"]], on="filename")
props.to_csv(OUT / "image_properties.csv", index=False)
summary["image_short_side"] = {"min": int(props.short_side.min()), "median": int(props.short_side.median()),
                               "max": int(props.short_side.max()),
                               "under_600px": int((props.short_side < 600).sum()),
                               "under_300px": int((props.short_side < 300).sum())}
summary["image_format_mode"] = (props.format + " " + props["mode"]).value_counts().to_dict()
summary["short_side_median_by_source"] = props.groupby("source").short_side.median().to_dict()

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(props.short_side, bins=np.arange(0, 4400, 200), color="#14213D")
ax.axvline(224, color="#B5521B", ls="--", label="model input 224 px")
ax.set(xlabel="Shorter image side (px)", ylabel="Images", title="Source resolution of the 94 images")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "resolution.png", dpi=160)

# ---------- ① 纳入标准：视角 ----------
# 人工逐张查看 94 张图后记录的烧录文字（2026-10-28），只列与“全部为 PA”相关的图。
review = pd.read_csv(OUT / "manual_view_review.csv")
changes = pd.read_csv(RESULTS / "metadata_changes.csv")
relabelled = set(changes.filename)
not_pa_marked = set(review.loc[review.burned_in_view_says_not_pa, "filename"])
summary["view_audit"] = {
    "relabelled_or_removed_in_current_metadata": len(relabelled),
    "burned_in_not_pa_but_still_pa_in_current_metadata": sorted(not_pa_marked - relabelled),
    "total_not_reliably_pa": len(relabelled | not_pa_marked),
}

# ---------- 误差是否随来源变化（Part I 单特征模型，50 次划分合并）----------
pred = pd.read_csv(RESULTS / "test_predictions.csv")
pred = pred[pred.method == "lung_opacity"].merge(cohort[["filename", "source"]], on="filename")
pred["abs_error"] = (pred.y_pred - pred.y_true).abs()
by_source = pred.groupby(["source", "target"]).abs_error.mean().unstack().round(3)
by_source["images"] = cohort.source.value_counts()
by_source.to_csv(OUT / "mae_by_source.csv")
summary["mae_by_source"] = by_source.to_dict("index")

# ---------- ② 示例图（原样导出，低分辨率用最近邻放大以显示像素）----------
EXAMPLES = {
    "example_ap_sitting.png": "ryct.2020200034.fig5-day4.jpeg",
    "example_pdf_arrow.png": "1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1a.png",
    "example_tiny_ap.png": "41591_2020_819_Fig1_HTML.webp-day10.png",
}
for name, f in EXAMPLES.items():
    with Image.open(IMAGES / f) as im:
        im = im.convert("L")
        scale = 720 / im.size[1]
        im.resize((round(im.size[0] * scale), 720), Image.NEAREST).save(FIG / name)

(OUT / "partC_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                        encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
