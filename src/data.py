"""Dataset download + loading. Idempotent: skips anything already on disk.

- Chest X-rays : NIH ChestX-ray14 via Parquet mirrors on HuggingFace
                 (the original `alkzar90/NIH-Chest-X-ray-dataset` uses a
                 loading script, which HF `datasets` v4.5+ dropped support
                 for; we use the auto-converted parquet mirrors instead).
- MRI slices   : MSD Task05 Prostate (pelvic/lower-abdominal T2 MRI).
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

from src.config import (
    DATA_DIR, MRI_DIR, MSD_ROOT, MSD_TAR, MSD_URL,
    N_MRI, N_XRAY, XRAY_DIR,
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


# -------------------------- MRI ----------------------------------------------

def _ensure_msd() -> None:
    if MSD_ROOT.exists():
        return
    if not MSD_TAR.exists():
        print(f"[data] downloading MSD Task05 Prostate -> {MSD_TAR}")
        urllib.request.urlretrieve(MSD_URL, MSD_TAR)
    print(f"[data] extracting {MSD_TAR}")
    subprocess.run(["tar", "-xf", str(MSD_TAR), "-C", str(DATA_DIR)], check=True)


def download_mri(n: int = N_MRI) -> List[Path]:
    existing = sorted(MRI_DIR.glob("mri_*.png"))
    if len(existing) >= n:
        print(f"[data] re-using {len(existing)} cached MRI slices")
        return existing[:n]

    _ensure_msd()
    nii_files = sorted((MSD_ROOT / "imagesTr").glob("prostate_*.nii.gz"))[:n]
    paths = []
    for i, nii in enumerate(nii_files):
        vol = nib.load(str(nii)).get_fdata()
        if vol.ndim == 4: vol = vol[..., 0]                   # T2 channel
        sl = vol[:, :, vol.shape[2] // 2]
        lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
        sl = np.clip(sl, lo, hi)
        sl = ((sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * 255).astype(np.uint8)
        sl = np.rot90(sl)
        p  = MRI_DIR / f"mri_{i:02d}.png"
        Image.fromarray(sl).convert("L").save(p)
        paths.append(p)
    print(f"[data] saved {len(paths)} MRI slices -> {MRI_DIR}")
    return paths


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
