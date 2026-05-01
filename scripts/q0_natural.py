"""Q1 natural-image baseline.

The PDF spec for Q1 asks how SAM's intermediate representations look
"compared to the natural image" — but our main analysis only contrasts
X-ray with MRI. This script anchors that comparison directly: it runs
~10 natural photos through the same encoder, captures per-block attention
entropy, token norms, and pooled features, and produces a three-trace
overlay plot (Natural / X-ray / MRI). It also reports cross-CKA between
the natural baseline and each medical modality at every depth.

Reads cached `outputs/q1_features.npz` for the X-ray and MRI side; the
natural side requires fresh forward passes (~30 sec on T4 for 10 images).

Outputs:
    figures/fig9_natural_baseline.{pdf,png}
    outputs/natural_layerwise.npz
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src import config
from src.data import download_natural, load_natural
from src.metrics import linear_cka
from src.model import clear_cuda, encode, load_sam, set_store
from src.viz import PALETTE, apply_pub_style, save_fig


def _pooled(tok: torch.Tensor) -> torch.Tensor:
    return tok.reshape(-1, tok.shape[-1]).mean(0)


def _load_medical_pooled():
    """Return pooled per-block features cached from scripts.q1, plus labels.
    Also returns per-image diagnostic arrays (norms, entropy) when present."""
    p = config.OUT_DIR / "q1_features.npz"
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Run `python -m scripts.q1` first.")
    d = np.load(p, allow_pickle=True)
    labels = d["labels"]
    blocks = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("block_"))
    feats  = {b: d[f"block_{b}"] for b in blocks}
    extra  = dict(
        token_norms = d["token_norms_by_block"] if "token_norms_by_block" in d.files else None,
        entropy     = d["entropy_by_block"]     if "entropy_by_block"     in d.files else None,
    )
    return feats, labels, blocks, extra


def main() -> None:
    apply_pub_style()
    download_natural(config.N_NATURAL)
    _, _, meta = load_sam()
    block_ids = list(range(meta["n_blocks"]))
    global_attn = meta["global_attn"]

    # ---- forward pass through the encoder, capture per-block stats ----
    images = load_natural()
    if not images:
        raise RuntimeError("No natural images on disk after download_natural().")

    pooled_nat   = {b: [] for b in block_ids}
    norm_nat_per = {b: [] for b in block_ids}
    ent_nat_per  = {b: [] for b in block_ids}

    set_store(entropy=True)
    for img, _ in tqdm(images, desc="q0_natural/encoding"):
        out = encode(img, block_ids=block_ids)
        for b in block_ids:
            tok = out["tokens"][b][0]
            pooled_nat  [b].append(_pooled(tok))
            norm_nat_per[b].append(tok.flatten(0, 1).norm(dim=-1).mean().item())
            ent = out["attn_entropy"].get(b)
            ent_nat_per[b].append(ent.mean().item() if ent is not None else np.nan)
        del out; clear_cuda()
    set_store(entropy=False)

    pooled_nat = {b: torch.stack(v).numpy() for b, v in pooled_nat.items()}

    # ---- pull X-ray and MRI from cache for the comparison ----
    feats_med, labels, blocks, med_extra = _load_medical_pooled()
    xi = np.where(labels == "xray")[0]
    mi = np.where(labels == "mri" )[0]

    # CKA needs equal-size sample sets on both sides (it compares N×N Gram
    # matrices). Natural set has 10; subsample 10 X-rays and 10 MRIs.
    n_nat = len(images)
    rng = np.random.default_rng(config.SEED)
    xi_sub = rng.choice(xi, size=n_nat, replace=False)
    mi_sub = rng.choice(mi, size=n_nat, replace=False)

    rows = []
    for b in blocks:
        rows.append(dict(
            block=b,
            norm_nat     = float(np.mean(norm_nat_per[b])),
            cka_nat_xray = linear_cka(pooled_nat[b], feats_med[b][xi_sub]),
            cka_nat_mri  = linear_cka(pooled_nat[b], feats_med[b][mi_sub]),
        ))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- per-block per-image diagnostics for medical (from q1 cache) ----
    norms_med = med_extra["token_norms"]    # (n_blocks, n_imgs) or None
    ent_med   = med_extra["entropy"]        # (n_blocks, n_imgs) or None
    have_med_diag = norms_med is not None and ent_med is not None
    if have_med_diag:
        norms_xray_b = norms_med[:, xi].mean(axis=1)
        norms_mri_b  = norms_med[:, mi].mean(axis=1)
        ent_xray_b   = np.nanmean(ent_med[:, xi], axis=1)
        ent_mri_b    = np.nanmean(ent_med[:, mi], axis=1)
    else:
        print("[q0_natural] note: q1_features.npz lacks per-image norm/entropy "
              "arrays. Re-run scripts.q1 with the latest code to enable the "
              "3-modality overlay in fig9.")

    # ---- Fig 9: 3-panel overlay (Natural / X-ray / MRI) ----
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6),
                             gridspec_kw=dict(wspace=0.40))

    # natural-side per-block stats
    ent_nat_b  = np.array([float(np.nanmean(ent_nat_per[b]))  for b in blocks])
    norm_nat_b = np.array([float(np.mean(norm_nat_per[b]))    for b in blocks])

    # (a) attention entropy at global blocks — Natural vs X-ray vs MRI
    ax = axes[0]
    gab_idx = [blocks.index(b) for b in global_attn]
    ax.plot(global_attn, ent_nat_b[gab_idx], "-o",
            color=PALETTE["accent"], markersize=6, linewidth=2.0, label="Natural")
    if have_med_diag:
        ax.plot(global_attn, ent_xray_b[gab_idx], "-o",
                color=PALETTE["xray"], markersize=6, linewidth=2.0, label="X-ray")
        ax.plot(global_attn, ent_mri_b[gab_idx], "-o",
                color=PALETTE["mri"],  markersize=6, linewidth=2.0, label="MRI")
    ax.set_xlabel("Global-attn block")
    ax.set_ylabel("Attention entropy (nats)")
    ax.set_title("(a) Attention entropy", loc="left")
    ax.set_xticks(global_attn); ax.legend(loc="best")

    # (b) token-norm growth — three traces overlaid
    ax = axes[1]
    ax.plot(blocks, norm_nat_b, "-o", color=PALETTE["accent"],
            markersize=5, linewidth=1.8, label="Natural")
    if have_med_diag:
        ax.plot(blocks, norms_xray_b, "-o", color=PALETTE["xray"],
                markersize=5, linewidth=1.8, label="X-ray")
        ax.plot(blocks, norms_mri_b, "-o", color=PALETTE["mri"],
                markersize=5, linewidth=1.8, label="MRI")
    ax.set_xlabel("Encoder block")
    ax.set_ylabel(r"mean per-token $\|\mathbf{z}\|_2$")
    ax.set_title("(b) Token-norm growth", loc="left")
    ax.set_xticks(blocks); ax.legend(loc="best")

    # (c) cross-CKA: how geometrically similar is "natural" to each medical modality?
    ax = axes[2]
    ax.plot(blocks, summary["cka_nat_xray"].values, "-o", color=PALETTE["xray"],
            markersize=5, linewidth=1.8, label="CKA(Natural, X-ray)")
    ax.plot(blocks, summary["cka_nat_mri"].values,  "-o", color=PALETTE["mri"],
            markersize=5, linewidth=1.8, label="CKA(Natural, MRI)")
    ax.set_xlabel("Encoder block")
    ax.set_ylabel("CKA")
    ax.set_title("(c) Geometric similarity to natural baseline", loc="left")
    ax.set_xticks(blocks); ax.legend(loc="best")
    ax.set_ylim(0, max(0.5, summary[["cka_nat_xray","cka_nat_mri"]].values.max() + 0.05))

    fig.suptitle(
        "Natural-image baseline: how SAM treats X-ray and MRI relative to "
        "ordinary photographs.",
        y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig9_natural_baseline")
    plt.close(fig)

    # ---- dump for downstream / report ----
    np.savez(config.OUT_DIR / "natural_layerwise.npz",
             blocks=np.array(blocks),
             pooled=np.stack([pooled_nat[b] for b in blocks]),     # (B, n, D)
             norms =np.array([np.mean(norm_nat[b]) for b in blocks]),
             cka_nat_xray=summary["cka_nat_xray"].values,
             cka_nat_mri =summary["cka_nat_mri"].values)
    summary.to_csv(config.OUT_DIR / "natural_summary.csv", index=False)
    print(f"[q0_natural] done -> figures/fig9_natural_baseline.pdf, "
          f"outputs/natural_layerwise.npz")


if __name__ == "__main__":
    main()
