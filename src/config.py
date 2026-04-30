"""Paths, seeds, device, and experiment constants.

Override the output root by exporting SAM_INTERP_ROOT before running any
script (useful locally; on Colab the default /content/sam_interp is fine).
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT      = Path(os.environ.get("SAM_INTERP_ROOT", "/content/sam_interp"))
DATA_DIR  = ROOT / "data"
CKPT_DIR  = ROOT / "checkpoints"
FIG_DIR   = ROOT / "figures"
OUT_DIR   = ROOT / "outputs"       # csv / pt dumps
for _p in (ROOT, DATA_DIR, CKPT_DIR, FIG_DIR, OUT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# -- model --
SAM_VARIANT   = "vit_b"
SAM_CKPT_URL  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CKPT_PATH = CKPT_DIR / "sam_vit_b_01ec64.pth"
PATCH_HW      = 64                 # 1024 / 16

# -- data --
N_XRAY   = 20
N_MRI    = 20

# NIH X-ray mirror is resolved at runtime (see src/data.py). No URL here.

# CHAOS abdominal MRI (Combined Healthy Abdominal Organ Segmentation) — public
# Zenodo mirror of the MICCAI CHAOS challenge. 20 cases × T2-SPIR DICOMs; we
# take one mid-axial slice per case → 20 abdominal MRI slices.
CHAOS_URL  = "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1"
CHAOS_ZIP  = DATA_DIR / "CHAOS_Train_Sets.zip"
CHAOS_ROOT = DATA_DIR / "CHAOS_Train_Sets"

XRAY_DIR    = DATA_DIR / "xray"
MRI_DIR     = DATA_DIR / "mri"
NATURAL_DIR = DATA_DIR / "natural"          # ImageNet-style baseline for Q1
XRAY_DIR   .mkdir(exist_ok=True)
MRI_DIR    .mkdir(exist_ok=True)
NATURAL_DIR.mkdir(exist_ok=True)
N_NATURAL  = 10
