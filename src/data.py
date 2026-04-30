"""Dataset download + loading. Idempotent: skips anything already on disk.

- Chest X-rays : NIH ChestX-ray14 via Parquet mirrors on HuggingFace
                 (the original `alkzar90/NIH-Chest-X-ray-dataset` uses a
                 loading script, which HF `datasets` v4.5+ dropped support
                 for; we use the auto-converted parquet mirrors instead).
- MRI slices   : CHAOS T2-SPIR abdominal MRI (Zenodo mirror, no signup).
                 20 cases × one mid-axial slice each.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.config import (
    CHAOS_ROOT, CHAOS_URL, CHAOS_ZIP, DATA_DIR, MRI_DIR,
    N_MRI, N_NATURAL, N_XRAY, NATURAL_DIR, XRAY_DIR,
)


# -------------------------- chest X-rays -------------------------------------

# NIH ChestX-ray14 parquet mirrors (no loading scripts → work with HF datasets
# v4.5+). The "small" mirror is 7.3k images / 3 GB total; streaming only pulls
# the first ~20 before we break out, so bandwidth is trivial.
_XRAY_PARQUET_MIRRORS = [
    ("Sohaibsoussi/NIH-Chest-X-ray-dataset-small", "train"),   # ~3 GB total
    ("BahaaEldin0/NIH-Chest-Xray-14",              "train"),   # full 112k / 45 GB
]


def _try_stream(repo: str, split: str, n: int) -> List[Path]:
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    paths = []
    for i, item in enumerate(ds):
        if i >= n: break
        img = item.get("image") or item.get("img") or item.get("pixel_values")
        if img is None:
            raise KeyError(f"no image field in {repo}; keys: {list(item.keys())}")
        if img.mode not in ("L", "RGB"): img = img.convert("L")
        p = XRAY_DIR / f"xray_{i:02d}.png"
        img.save(p); paths.append(p)
    return paths


def download_xrays(n: int = N_XRAY) -> List[Path]:
    """Stream `n` NIH ChestX-ray14 images from the first working parquet mirror."""
    existing = sorted(XRAY_DIR.glob("xray_*.png"))
    if len(existing) >= n:
        print(f"[data] re-using {len(existing)} cached X-rays")
        return existing[:n]

    last_err = None
    for repo, split in _XRAY_PARQUET_MIRRORS:
        try:
            print(f"[data] streaming NIH X-rays from {repo} (split={split})")
            paths = _try_stream(repo, split, n)
            if len(paths) >= n:
                print(f"[data] saved {len(paths)} NIH X-rays from {repo} -> {XRAY_DIR}")
                return paths
        except Exception as e:                       # noqa: BLE001
            last_err = e
            print(f"[data]   skipped ({type(e).__name__}: {e})")

    raise RuntimeError(
        "All NIH X-ray mirrors failed. Fallback: drop 10–30 chest X-ray PNG "
        f"files into {XRAY_DIR} and rerun — load_all() will pick them up. "
        f"Last error: {last_err}")


# -------------------------- abdominal MRI (CHAOS T2-SPIR) --------------------

def _ensure_chaos() -> None:
    """Download + extract the CHAOS training set if not already on disk."""
    if CHAOS_ROOT.exists():
        return
    if not CHAOS_ZIP.exists():
        print(f"[data] downloading CHAOS_Train_Sets ({CHAOS_URL}) -> {CHAOS_ZIP}")
        urllib.request.urlretrieve(CHAOS_URL, CHAOS_ZIP)
    print(f"[data] extracting {CHAOS_ZIP}  (~2 GB disk)")
    CHAOS_ROOT.mkdir(exist_ok=True)
    with zipfile.ZipFile(CHAOS_ZIP) as z:
        z.extractall(CHAOS_ROOT)


def _load_dicom_slice(dcm_path: Path) -> np.ndarray:
    """Read a DICOM file, return uint8 grayscale image. pydicom is lazy-imported
    so the package only becomes a hard dep when MRI is actually downloaded."""
    import pydicom
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).astype(np.uint8)


def download_mri(n: int = N_MRI) -> List[Path]:
    """Take one mid-axial T2-SPIR slice from each of the first `n` CHAOS cases.

    CHAOS has 20 MRI cases, so n is clamped to 20. Each case's DICOM folder
    holds ~26–40 axial slices; we grab the middle one per case to get 20
    diverse, well-centered abdominal views.
    """
    existing = sorted(MRI_DIR.glob("mri_*.png"))
    if len(existing) >= n:
        print(f"[data] re-using {len(existing)} cached MRI slices")
        return existing[:n]

    _ensure_chaos()
    # Zenodo bundles sometimes nest as Train_Sets/MR, sometimes as just MR;
    # find the "MR" directory anywhere under CHAOS_ROOT.
    mr_root = next((p for p in CHAOS_ROOT.rglob("MR") if p.is_dir()), None)
    if mr_root is None:
        raise RuntimeError(
            f"CHAOS MR folder not found anywhere under {CHAOS_ROOT}. "
            "Extraction may have failed — check the zip contents.")

    cases = sorted([p for p in mr_root.iterdir() if p.is_dir()])
    n = min(n, len(cases))
    paths = []
    for i, case in enumerate(cases[:n]):
        dcm_dir = case / "T2SPIR" / "DICOM_anon"
        dcms = sorted(dcm_dir.glob("*.dcm"))
        if not dcms:
            print(f"[data]   case {case.name}: no DICOMs in {dcm_dir}, skipping")
            continue
        mid = dcms[len(dcms) // 2]
        img = _load_dicom_slice(mid)
        p = MRI_DIR / f"mri_{i:02d}.png"
        Image.fromarray(img).convert("L").save(p)
        paths.append(p)

    print(f"[data] saved {len(paths)} CHAOS T2-SPIR abdominal MRI slices -> {MRI_DIR}")
    return paths


# -------------------------- natural-image baseline (Q1) ----------------------

# Direct HTTPS download from the public COCO val2017 server. COCO has been
# hosted at this URL pattern continuously since 2017 and requires no auth.
# Each image is a real natural photograph (objects, people, scenes), which
# is exactly what we want as a Q1 anchor against SAM's SA-1B pretraining.
_COCO_VAL2017_BASE = "http://images.cocodataset.org/val2017/{:012d}.jpg"
_NATURAL_COCO_IDS = [
    139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
    1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761,
]


def download_natural(n: int = N_NATURAL) -> List[Path]:
    """Download `n` natural photographs from COCO val2017 (public, no auth)."""
    existing = sorted(NATURAL_DIR.glob("nat_*.png"))
    if len(existing) >= n:
        print(f"[data] re-using {len(existing)} cached natural images")
        return existing[:n]

    paths = []
    for i, coco_id in enumerate(_NATURAL_COCO_IDS):
        if len(paths) >= n: break
        url = _COCO_VAL2017_BASE.format(coco_id)
        out = NATURAL_DIR / f"nat_{i:02d}.png"
        try:
            print(f"[data] fetching {url}")
            urllib.request.urlretrieve(url, out.with_suffix(".jpg"))
            Image.open(out.with_suffix(".jpg")).convert("RGB").save(out)
            out.with_suffix(".jpg").unlink()
            paths.append(out)
        except Exception as e:                       # noqa: BLE001
            print(f"[data]   skipped {url} ({type(e).__name__}: {e})")

    if len(paths) < n:
        raise RuntimeError(
            f"Only got {len(paths)} of {n} natural images from COCO. "
            f"Fallback: drop ≥{n} photo PNGs into {NATURAL_DIR} and rerun.")
    print(f"[data] saved {len(paths)} natural images -> {NATURAL_DIR}")
    return paths


def load_natural() -> List[Tuple[Image.Image, str]]:
    paths = sorted(NATURAL_DIR.glob("nat_*.png"))
    return [(Image.open(p), "natural") for p in paths]


# -------------------------- unified loader -----------------------------------

def load_all() -> List[Tuple[Image.Image, str]]:
    """Return the combined [(PIL, label)] list, X-rays first then MRIs."""
    xray_paths = sorted(XRAY_DIR.glob("xray_*.png"))
    mri_paths  = sorted(MRI_DIR.glob("mri_*.png"))
    if not xray_paths or not mri_paths:
        raise RuntimeError(
            "No cached images. Run `python -m scripts.download` first.")
    items  = [(Image.open(p), "xray") for p in xray_paths]
    items += [(Image.open(p), "mri")  for p in mri_paths]
    return items
