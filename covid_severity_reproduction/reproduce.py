"""Reproduce Cohen et al. (2020), Table 1, using the author's protocol.

Run `python reproduce.py prepare`, then `python reproduce.py run`.
The historical data snapshot is a documented reconstruction, not an author-pinned release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SOURCES = ROOT / "sources"
DATA_COMMIT = "d929fa7ba8ded7aa5233cbd23b7685f3677592db"
SCORE_COMMIT = "78543292f8b01d5e0ed1d0e15dce71949f0657bb"
CODE_COMMIT = "5571e45b49bdba69161268f8bfaea90d698f9f55"
WEIGHT_URL = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)
WEIGHT_FILE = ROOT / "weights" / WEIGHT_URL.rsplit("/", 1)[-1]
METHODS = ["lung_opacity", "four_logits", "eighteen_logits", "intermediate_features", "constant"]
METRICS = ["pearson_r", "r2", "mae", "mse"]
PACKAGES = ["torch", "torchvision", "torchxrayvision", "numpy", "pandas", "scipy", "scikit-learn", "scikit-image", "matplotlib", "pillow"]
# Values transcribed from Table 1, page 4 of the published Cureus paper.
# Each pair is mean, sample standard deviation; baseline Pearson=0 is a display convention.
PAPER = {
    "opacity": [
        [(0.78, 0.04), (0.58, 0.09), (0.78, 0.05), (0.86, 0.11)],
        [(0.78, 0.04), (0.58, 0.09), (0.76, 0.05), (0.87, 0.12)],
        [(0.73, 0.09), (0.44, 0.16), (0.86, 0.11), (1.15, 0.33)],
        [(0.66, 0.08), (0.25, 0.21), (1.01, 0.09), (1.54, 0.28)],
        [(0.00, 0.00), (-0.08, 0.10), (1.24, 0.10), (2.26, 0.36)],
    ],
    "extent": [
        [(0.80, 0.05), (0.60, 0.09), (1.14, 0.11), (2.06, 0.34)],
        [(0.79, 0.05), (0.57, 0.10), (1.19, 0.11), (2.17, 0.37)],
        [(0.76, 0.08), (0.47, 0.16), (1.32, 0.17), (2.73, 0.89)],
        [(0.74, 0.08), (0.43, 0.16), (1.36, 0.13), (2.88, 0.58)],
        [(0.00, 0.00), (-0.08, 0.10), (2.00, 0.17), (5.60, 0.95)],
    ],
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Cohen2020-reproduction"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if not temporary.stat().st_size:
                raise ValueError(f"Empty download: {url}")
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/{DATA_COMMIT}"
    assets = [
        (base + "/metadata.csv", DATA / "metadata.csv"),
        (f"https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/{SCORE_COMMIT}/annotations/covid-severity-scores.csv", DATA / "severity_scores.csv"),
        (WEIGHT_URL, WEIGHT_FILE),
    ]
    for url, destination in assets:
        print(f"Preparing {destination.name}", flush=True)
        download(url, destination)
    lines = (DATA / "severity_scores.csv").read_text(encoding="utf-8").splitlines()
    scores = list(csv.DictReader(lines[5:]))
    if len(scores) != 94 or len({r["filename"] for r in scores}) != 94:
        raise ValueError("The paper requires 94 unique annotated images.")
    for row in scores:
        filename = row["filename"]
        if Path(filename).name != filename or "\\" in filename:
            raise ValueError(f"Unexpected image filename: {filename}")
        assets.append((base + "/images/" + urllib.parse.quote(filename), DATA / "images" / filename))
    image_assets = assets[3:]
    with ThreadPoolExecutor(max_workers=6) as executor:
        for i, _ in enumerate(executor.map(lambda asset: download(*asset), image_assets), 1):
            if i % 10 == 0 or i == 94:
                print(f"Images ready: {i}/94", flush=True)
    if WEIGHT_FILE.stat().st_size != 28_382_008:
        raise ValueError("Official pretrained weight size does not match its release asset.")
    manifest = [
        {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "url": url,
         "bytes": path.stat().st_size, "sha256": sha256(path)}
        for url, path in assets
    ]
    write_json(DATA / "download_manifest.json", manifest)
    print("Prepared 94 images, historical metadata, scores, and original March 2020 weights.", flush=True)


def load_cohort():
    import numpy as np
    import pandas as pd

    for entry in json.loads((DATA / "download_manifest.json").read_text(encoding="utf-8")):
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Downloaded file changed: {entry['path']}")
    metadata = pd.read_csv(DATA / "metadata.csv")
    scores = pd.read_csv(DATA / "severity_scores.csv", skiprows=5)
    if scores["filename"].duplicated().any():
        raise ValueError("Duplicate score filenames would make the annotation join ambiguous.")
    # Preserve historical metadata order and numeric patient IDs, as in the author's loader.
    pa = metadata.loc[(metadata["view"] == "PA") & ~metadata["finding"].isin(["todo", "Unknown"])]
    if pa["filename"].duplicated().any():
        raise ValueError("Duplicate PA filenames would make the annotation join ambiguous.")
    cohort = pa.merge(scores, how="left", on="filename", sort=False, validate="one_to_one")
    cohort = cohort.loc[cohort["geographic_mean"].notna()].reset_index(drop=True)
    assert len(cohort) == 94 and set(cohort.filename) == set(scores.filename)
    assert cohort.patientid.notna().all() and cohort.patientid.nunique() == 65
    assert cohort.view.eq("PA").all()
    assert cohort.geographic_mean.between(0, 8).all() and cohort.opacity_mean.between(0, 6).all()
    assert np.issubdtype(cohort.patientid.dtype, np.number), "Do not change numeric group sorting."
    cohort.to_csv(RESULTS / "cohort.csv", index=False)
    write_json(RESULTS / "data_audit.json", {
        "snapshot_commit": DATA_COMMIT, "snapshot_PA_count": len(pa),
        "annotated_images": len(cohort), "unique_patients": int(cohort.patientid.nunique()),
        "patient_id_dtype": str(cohort.patientid.dtype),
        "missing_scores_or_images": 0, "all_views_PA_in_historical_metadata": True,
        "note": "Historical snapshot chosen to match the saved notebook's 199 PA / 94 scored images; author did not pin an exact data commit. Later metadata revises 11 scored views, removes one scored image and changes inferred patient ID dtype; IDs of the 93 common images are unchanged.",
    })
    return cohort


def extract_features(cohort, environment):
    import numpy as np
    import skimage.io
    import torch
    import torch.nn.functional as F
    import torchxrayvision as xrv

    fingerprint = {
        "script_sha256": sha256(__file__), "downloads_sha256": sha256(DATA / "download_manifest.json"),
        "cohort_sha256": sha256(RESULTS / "cohort.csv"), "packages": environment["packages"],
    }
    cache = RESULTS / "features.npz"
    cache_metadata = RESULTS / "features_manifest.json"
    if cache.exists() and cache_metadata.exists() and json.loads(cache_metadata.read_text(encoding="utf-8")) == fingerprint:
        with np.load(cache, allow_pickle=False) as arrays:
            return arrays["features"], arrays["logits"]
    # Official trusted release is an old serialized model, rather than a plain state dict.
    saved = torch.load(WEIGHT_FILE, map_location="cpu", weights_only=False)
    for module in saved.modules():
        if not hasattr(module, "_non_persistent_buffers_set"):
            module._non_persistent_buffers_set = set()
    model = xrv.models.DenseNet(weights=None)
    model.load_state_dict(saved.state_dict())
    model.op_threshs = None
    model.apply_sigmoid = False
    model.eval().requires_grad_(False)
    crop, resize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
    images = []
    for filename in cohort.filename:
        image = skimage.io.imread(DATA / "images" / filename)
        if image.ndim == 3:
            image = image[:, :, 0]  # Match 2020 loader, not RGB luminance conversion.
        assert image.ndim == 2 and image.max() <= 255
        image = (2 * (image.astype(np.float32) / 255) - 1) * 1024
        images.append(resize(crop(image[None, :, :])))
    images = np.stack(images)
    assert images.shape == (94, 1, 224, 224) and np.isfinite(images).all()
    features, logits = [], []
    with torch.inference_mode():
        for start in range(0, 94, 16):
            batch = torch.from_numpy(images[start:start + 16])
            pooled = F.adaptive_avg_pool2d(F.relu(model.features(batch), inplace=True), (1, 1)).flatten(1)
            output = model.classifier(pooled)
            if start == 0:
                torch.testing.assert_close(output[:1], model(batch[:1]), rtol=1e-4, atol=1e-4)
            features.append(pooled.numpy())
            logits.append(output.numpy())
            print(f"Features extracted: {min(start + 16, 94)}/94", flush=True)
    features, logits = np.concatenate(features), np.concatenate(logits)
    assert features.shape == (94, 1024) and logits.shape == (94, 18)
    assert np.isfinite(features).all() and np.isfinite(logits).all()
    np.savez_compressed(cache, features=features, logits=logits)
    write_json(cache_metadata, fingerprint)
    return features, logits


def evaluate(cohort, features, logits):
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    import torchxrayvision as xrv

    columns = xrv.datasets.default_pathologies
    four = [columns.index(p) for p in ["Lung Opacity", "Pneumonia", "Infiltration", "Consolidation"]]
    assert four == [16, 8, 2, 1]
    inputs = [logits[:, [four[0]]], logits[:, four], logits, features, np.zeros((94, 1))]
    groups = cohort.patientid.to_numpy()
    scores, predictions, splits = [], [], []
    for seed in range(50):
        splitter = GroupShuffleSplit(train_size=0.5, test_size=0.5, random_state=seed)
        train, test = next(splitter.split(inputs[0], groups=groups))
        assert set(groups[train]).isdisjoint(groups[test]), "Patient leakage!"
        assert len(train) + len(test) == 94
        splits.append({"seed": seed, "train_indices": train.tolist(), "test_indices": test.tolist(),
                       "train_patients": np.unique(groups[train]).tolist(), "test_patients": np.unique(groups[test]).tolist()})
        for target, label in [("opacity", "opacity_mean"), ("extent", "geographic_mean")]:
            y = cohort[label].to_numpy(dtype=float)
            for method, x in zip(METHODS, inputs):
                model = LinearRegression().fit(x[train], y[train])
                predicted = model.predict(x[test])
                if method == "constant":
                    np.testing.assert_allclose(predicted, y[train].mean())
                r = 0.0 if method == "constant" else float(pearsonr(predicted, y[test]).statistic)
                scores.append({"target": target, "method": method, "seed": seed,
                               "n_train": len(train), "n_test": len(test), "pearson_r": r,
                               "r2": r2_score(y[test], predicted), "mae": mean_absolute_error(y[test], predicted),
                               "mse": mean_squared_error(y[test], predicted)})
                predictions.extend({"target": target, "method": method, "seed": seed,
                                    "filename": cohort.filename.iloc[index], "patientid": int(groups[index]),
                                    "y_true": float(y[index]), "y_pred": float(value)}
                                   for index, value in zip(test, predicted))
        if (seed + 1) % 10 == 0:
            print(f"Patient-grouped splits evaluated: {seed + 1}/50", flush=True)
    frame = pd.DataFrame(scores)
    assert len(frame) == 500 and np.isfinite(frame[METRICS].to_numpy()).all()
    frame.to_csv(RESULTS / "metrics_per_split.csv", index=False)
    pd.DataFrame(predictions).to_csv(RESULTS / "test_predictions.csv", index=False)
    write_json(RESULTS / "splits.json", splits)
    comparison = []
    for target in PAPER:
        for method, reference in zip(METHODS, PAPER[target]):
            values = frame.loc[(frame.target == target) & (frame.method == method)]
            row = {"target": target, "method": method}
            for metric, (paper_mean, paper_std) in zip(METRICS, reference):
                mean, std = float(values[metric].mean()), float(values[metric].std(ddof=1))
                row.update({f"{metric}_mean": mean, f"{metric}_std": std,
                            f"{metric}_paper_mean": paper_mean, f"{metric}_paper_std": paper_std,
                            f"{metric}_delta": mean - paper_mean})
            comparison.append(row)
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(RESULTS / "table1_comparison.csv", index=False)
    formatted = comparison[["target", "method"]].copy()
    for metric in METRICS:
        formatted[metric] = comparison.apply(lambda row: f"{row[metric + '_mean']:.2f} +/- {row[metric + '_std']:.2f}", axis=1)
    formatted.to_csv(RESULTS / "table1.csv", index=False)
    print(formatted.to_string(index=False), flush=True)
    return frame, pd.DataFrame(predictions), comparison


def plot_results(frame, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, target, upper in zip(axes, ["opacity", "extent"], [6, 8]):
        selected = predictions.loc[(predictions.target == target) & (predictions.method == "lung_opacity") & (predictions.seed == 1)]
        ax.scatter(selected.y_true, selected.y_pred, s=28, alpha=0.75)
        ax.plot([0, upper], [0, upper], "--", color="gray")
        ax.set(xlabel="Expert mean score", ylabel="Predicted score", title=f"{target.capitalize()}: held-out patients, seed 1")
        ax.grid(alpha=0.15)
    fig.savefig(RESULTS / "heldout_scatter.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, target in zip(axes, ["opacity", "extent"]):
        values = [frame.loc[(frame.target == target) & (frame.method == method), "mae"] for method in METHODS]
        ax.boxplot(values, labels=["1 logit", "4 logits", "18 logits", "1024 features", "Constant"])
        ax.set(title=f"{target.capitalize()}: 50 patient splits", ylabel="Held-out MAE")
        ax.tick_params(axis="x", rotation=20)
        for i, reference in enumerate(PAPER[target]):
            ax.scatter(i + 1, reference[2][0], marker="D", color="#d35400", s=35, label="Paper mean" if i == 0 else None, zorder=3)
        ax.legend()
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(RESULTS / "mae_comparison.png", dpi=180)
    plt.close(fig)


def run():
    import torch
    from threadpoolctl import threadpool_limits

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    environment = {"python": sys.version, "platform": platform.platform(), "device": "cpu",
                   "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
                   "code_commit": CODE_COMMIT, "data_commit": DATA_COMMIT, "score_commit": SCORE_COMMIT}
    write_json(RESULTS / "environment.json", environment)
    started = time.perf_counter()
    cohort = load_cohort()
    print(f"Validated cohort: {len(cohort)} images / {cohort.patientid.nunique()} patients", flush=True)
    features, logits = extract_features(cohort, environment)
    with threadpool_limits(limits=4):
        frame, predictions, comparison = evaluate(cohort, features, logits)
    plot_results(frame, predictions)
    write_json(RESULTS / "run_summary.json", {"elapsed_seconds": time.perf_counter() - started,
               "images": len(cohort), "patients": int(cohort.patientid.nunique()),
               "splits": 50, "targets": 2, "methods": 5, "fits": len(frame),
               "patient_leakage_checked_in_every_split": True,
               "baseline_pearson_display_convention": 0,
               "weight_sha256": sha256(WEIGHT_FILE)})
    print(f"Complete. Results saved in {RESULTS}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare()
    else:
        run()
