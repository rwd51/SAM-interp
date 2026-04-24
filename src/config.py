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
MSD_URL  = "https://msd-for-monai.s3.us-east-1.amazonaws.com/Task05_Prostate.tar"
MSD_TAR  = DATA_DIR / "Task05_Prostate.tar"
MSD_ROOT = DATA_DIR / "Task05_Prostate"
XRAY_DIR = DATA_DIR / "xray"
MRI_DIR  = DATA_DIR / "mri"
XRAY_DIR.mkdir(exist_ok=True)
MRI_DIR .mkdir(exist_ok=True)
