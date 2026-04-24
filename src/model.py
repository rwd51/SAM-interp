"""SAM ViT-B loader + attention-interpretability hooks.

Public API:
    load_sam()            -> (sam, encoder, meta_dict)
    preprocess_pil(img)   -> 1x3x1024x1024 normalised tensor on DEVICE
    encode(img, blocks)   -> dict of per-block artefacts (tokens, optional attn)
    set_store(attn, head_out, entropy)      -> enable/disable per-pass artefacts
    set_ablation(block, head)               -> zero-out one head in one block

The patched Attention.forward stores artefacts ONLY when the corresponding
flag is on, so a default encode() pass has essentially zero memory overhead
over the vanilla SAM forward pass.
"""
from __future__ import annotations

import gc
import urllib.request
from types import MethodType
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import (
    DEVICE, PATCH_HW, SAM_CKPT_PATH, SAM_CKPT_URL, SAM_VARIANT,
)


# -------------------------- private state ------------------------------------

_sam = None
_enc = None
_STORE = {"attn": False, "head_out": False, "entropy": False}


# -------------------------- public toggles -----------------------------------

def set_store(attn: bool = False, head_out: bool = False, entropy: bool = False) -> None:
    """Enable expensive per-pass artefact capture. Defaults are all False."""
    _STORE["attn"]     = attn
    _STORE["head_out"] = head_out
    _STORE["entropy"]  = entropy


def set_ablation(block_idx: int | None, head_idx: int | None) -> None:
    """Zero head `head_idx` of block `block_idx` in every subsequent forward
    pass. Either argument may be None to clear the ablation."""
    assert _enc is not None, "call load_sam() first"
    for i, blk in enumerate(_enc.blocks):
        if block_idx is not None and head_idx is not None and i == block_idx:
            blk.attn._ablate_heads = [head_idx]
        else:
            blk.attn._ablate_heads = None


def set_ablation_multi(block_idx: int, head_idxs: Iterable[int]) -> None:
    """Zero multiple heads in one block (used by the pair-ablation probe)."""
    assert _enc is not None, "call load_sam() first"
    for i, blk in enumerate(_enc.blocks):
        blk.attn._ablate_heads = list(head_idxs) if i == block_idx else None


# -------------------------- model loader -------------------------------------

def _download_ckpt() -> None:
    if SAM_CKPT_PATH.exists():
        return
    SAM_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model] downloading SAM ViT-B checkpoint -> {SAM_CKPT_PATH}")
    urllib.request.urlretrieve(SAM_CKPT_URL, SAM_CKPT_PATH)


def _patched_attn_forward(self, x):
    """Drop-in replacement for segment_anything Attention.forward.

    Respects the `_STORE` flags (attn / head_out / entropy) and `self._ablate_heads`.
    Shapes and numerics match the vanilla module exactly.
    """
    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                                           # B,nH,N,d
    attn = (q * self.scale) @ k.transpose(-2, -1)
    if getattr(self, "use_rel_pos", False):
        from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
        attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
    attn = attn.softmax(dim=-1)                                       # B,nH,N,N

    if _STORE["entropy"]:
        p = attn.clamp_min(1e-12)
        self._attn_entropy = (-(p * p.log()).sum(-1)).mean(dim=(0, 2)).detach().cpu()  # (nH,)
    else:
        self._attn_entropy = None

    self._attn = attn.detach().float().cpu() if _STORE["attn"] else None

    abl = getattr(self, "_ablate_heads", None)
    out = attn @ v                                                    # B,nH,N,d
    if abl:
        out = out.clone()
        for h_idx in abl:
            out[:, h_idx] = 0.0

    self._head_out = out.detach().float().cpu() if _STORE["head_out"] else None

    out = out.transpose(1, 2).reshape(B, H, W, -1)
    return self.proj(out)


def _install_taps(enc) -> None:
    for blk in enc.blocks:
        blk.attn._ablate_heads   = None
        blk.attn._attn           = None
        blk.attn._head_out       = None
        blk.attn._attn_entropy   = None
        blk.attn.forward = MethodType(_patched_attn_forward, blk.attn)


def load_sam():
    """Load SAM once, install taps, freeze parameters. Idempotent."""
    global _sam, _enc
    if _sam is not None:
        return _sam, _enc, _meta()

    _download_ckpt()
    from segment_anything import sam_model_registry
    _sam = sam_model_registry[SAM_VARIANT](checkpoint=str(SAM_CKPT_PATH))
    _sam.to(DEVICE).eval()
    for p in _sam.parameters():
        p.requires_grad_(False)
    _enc = _sam.image_encoder
    _install_taps(_enc)
    meta = _meta()
    print(f"[model] SAM-{SAM_VARIANT} loaded on {DEVICE}: "
          f"{meta['n_blocks']} blocks, dim={meta['embed_dim']}, heads={meta['n_heads']}, "
          f"global_attn={meta['global_attn']}")
    return _sam, _enc, meta


def _meta() -> dict:
    assert _enc is not None
    return dict(
        n_blocks    = len(_enc.blocks),
        embed_dim   = _enc.blocks[0].attn.qkv.in_features,
        n_heads     = _enc.blocks[0].attn.num_heads,
        patch_hw    = PATCH_HW,
        global_attn = list(_enc.global_attn_indexes),
    )


# -------------------------- preprocessing ------------------------------------

_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(1, 3, 1, 1)
_PIXEL_STD  = torch.tensor([58.395,  57.12,  57.375], device=DEVICE).view(1, 3, 1, 1)


def preprocess_pil(img_pil: Image.Image) -> torch.Tensor:
    """PIL -> 1x3x1024x1024 float tensor on DEVICE, SAM-normalised & padded."""
    from segment_anything.utils.transforms import ResizeLongestSide
    tr = ResizeLongestSide(1024)
    arr = np.array(img_pil.convert("RGB"))
    arr = tr.apply_image(arr)
    x = torch.as_tensor(arr, device=DEVICE).permute(2, 0, 1).float().unsqueeze(0)
    x = (x - _PIXEL_MEAN) / _PIXEL_STD
    h, w = x.shape[-2:]
    return F.pad(x, (0, 1024 - w, 0, 1024 - h))


# -------------------------- encode -------------------------------------------

class _BlockTap:
    def __init__(self, enc, block_ids):
        self.outs, self.handles = {}, []
        for i in block_ids:
            self.handles.append(enc.blocks[i].register_forward_hook(self._mk(i)))
    def _mk(self, i):
        def fn(mod, inp, out): self.outs[i] = out.detach().float().cpu()
        return fn
    def remove(self):
        for h in self.handles: h.remove()


@torch.no_grad()
def encode(img_pil: Image.Image, block_ids=None) -> dict:
    """Run one forward pass, return per-block artefacts (subject to set_store).

    Return keys:
        tokens[i]        - (1, 64, 64, D)
        attn[i]          - full attn, if set_store(attn=True)
        head_out[i]      - per-head outputs, if set_store(head_out=True)
        attn_entropy[i]  - (nH,) entropy, if set_store(entropy=True)
        feat             - final neck output (1, 256, 64, 64)
    """
    sam, enc, meta = load_sam()
    if block_ids is None:
        block_ids = list(range(meta["n_blocks"]))
    tap = _BlockTap(enc, block_ids)
    x   = preprocess_pil(img_pil)
    feat = enc(x)
    tokens   = {i: tap.outs[i] for i in block_ids}
    attn     = {i: enc.blocks[i].attn._attn         for i in block_ids if enc.blocks[i].attn._attn         is not None}
    head_out = {i: enc.blocks[i].attn._head_out     for i in block_ids if enc.blocks[i].attn._head_out     is not None}
    entropy  = {i: enc.blocks[i].attn._attn_entropy for i in block_ids if enc.blocks[i].attn._attn_entropy is not None}
    tap.remove()
    return dict(tokens=tokens, attn=attn, head_out=head_out,
                attn_entropy=entropy, feat=feat.detach().float().cpu())


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
