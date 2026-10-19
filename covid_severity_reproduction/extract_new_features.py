"""Part II: extract image features from newer pretrained models (RAD-DINO, BiomedCLIP).

Run from this folder with the reproduction environment:
    .\\.venv\\Scripts\\python.exe -m pip install "transformers>=4.40,<5" "open_clip_torch==2.23.0"
    .\\.venv\\Scripts\\python.exe .\\extract_new_features.py

The script reads the same 94 images in the same order as results/cohort.csv (the order
used for results/features.npz), so every feature file lines up row by row.
Each model is frozen and used only as a feature extractor, exactly like the paper's DenseNet.
Outputs go to results/part2_features/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "part2_features"
RAD_DINO_ID = "microsoft/rad-dino"
BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_images(filenames):
    """Read each image as the 2020 loader did (first channel), then copy it to 3 channels."""
    import numpy as np
    from PIL import Image

    images = []
    for filename in filenames:
        with Image.open(DATA / "images" / filename) as handle:
            array = np.asarray(handle)
        if array.ndim == 3:
            array = array[:, :, 0]  # same choice as reproduce.py: first channel, not luminance
        if array.dtype != np.uint8:
            if array.max() > 255:
                raise ValueError(f"{filename}: pixel values above 255, unexpected for this dataset")
            array = array.astype(np.uint8)
        if array.ndim != 2:
            raise ValueError(f"{filename}: unexpected image shape {array.shape}")
        images.append(Image.fromarray(np.ascontiguousarray(np.stack([array] * 3, axis=-1))))
    return images


def rad_dino_encoder():
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(RAD_DINO_ID)
    model = AutoModel.from_pretrained(RAD_DINO_ID).eval().requires_grad_(False)

    def encode(batch):
        inputs = processor(images=batch, return_tensors="pt")
        output = model(**inputs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return pooled  # CLS embedding after the final layer norm

    return encode


def biomedclip_encoder():
    import torch
    from open_clip import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(BIOMEDCLIP_ID)
    model = model.eval().requires_grad_(False)

    def encode(batch):
        pixels = torch.stack([preprocess(image) for image in batch])
        return model.encode_image(pixels)  # projected image embedding, not normalised

    return encode


ENCODERS = {"raddino": (RAD_DINO_ID, rad_dino_encoder), "biomedclip": (BIOMEDCLIP_ID, biomedclip_encoder)}


def extract(name, images, filenames, batch_size):
    import numpy as np
    import torch

    model_id, build = ENCODERS[name]
    print(f"[{name}] loading {model_id} (first run downloads the weights)", flush=True)
    started = time.perf_counter()
    encode = build()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            chunks.append(encode(images[start:start + batch_size]).float().numpy())
            print(f"[{name}] features extracted: {min(start + batch_size, len(images))}/{len(images)}", flush=True)
        repeat = encode(images[:2]).float().numpy()
    features = np.concatenate(chunks).astype(np.float32)
    if features.shape[0] != len(filenames) or features.ndim != 2:
        raise ValueError(f"[{name}] unexpected feature shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"[{name}] features contain NaN or infinity")
    np.testing.assert_allclose(repeat, features[:2], rtol=1e-4, atol=1e-4,
                               err_msg=f"[{name}] the same image gave different features twice")
    path = OUT / f"features_{name}.npz"
    np.savez_compressed(path, features=features, filenames=np.array(filenames))
    seconds = time.perf_counter() - started
    print(f"[{name}] saved {path.relative_to(ROOT)}: shape {features.shape}, {seconds:.0f} s", flush=True)
    return {"model_id": model_id, "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "shape": list(features.shape), "seconds": round(seconds, 1), "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=sorted(ENCODERS), default=sorted(ENCODERS))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import pandas as pd
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(RESULTS / "cohort.csv")
    filenames = cohort["filename"].tolist()
    if len(filenames) != 94 or len(set(filenames)) != 94:
        raise ValueError("results/cohort.csv must list the 94 scored images; run reproduce.py first")
    images = load_images(filenames)
    print(f"Loaded {len(images)} images in cohort.csv order", flush=True)

    manifest_path = OUT / "features_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["environment"] = {
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: version(p) for p in ["torch", "transformers", "open_clip_torch", "timm", "numpy", "pillow"]},
    }
    manifest["cohort_sha256"] = sha256(RESULTS / "cohort.csv")
    failures = {}
    for name in args.models:
        try:
            manifest[name] = extract(name, images, filenames, args.batch_size)
        except Exception as error:  # keep going so one missing package does not block the other model
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"[{name}] FAILED: {failures[name]}", flush=True)
    manifest["failures"] = failures
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        sys.exit(1)
    print("Done. Next: .\\.venv\\Scripts\\python.exe .\\compare_features.py", flush=True)


if __name__ == "__main__":
    main()
