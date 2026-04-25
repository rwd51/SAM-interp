"""Q1 — Layer-wise CKA analysis (representational similarity).

Produces Fig. 8: three 12 × 12 linear-CKA heatmaps over all encoder blocks —
    (a) X-ray self-similarity     CKA(xray_i,  xray_j)
    (b) MRI   self-similarity     CKA(mri_i,   mri_j)
    (c) Cross-modality            CKA(xray_i,  mri_j)
plus the cross-modality diagonal as a line plot below (the "modality drift"
trace — at what depth do X-ray and MRI representations look most alike?).

Reads pooled per-block features from outputs/q1_features.npz, so this is a
post-processing script that requires `python -m scripts.q1` to have run first.

Output:
    figures/fig8_layerwise_cka.{pdf,png}
    outputs/q8_cka.npz                  (3 12x12 matrices for downstream use)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from src import config
from src.metrics import linear_cka
from src.viz import PALETTE, apply_pub_style, save_fig


def _load_features():
    p = config.OUT_DIR / "q1_features.npz"
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Run `python -m scripts.q1` first.")
    data = np.load(p, allow_pickle=True)
    labels = data["labels"]
    blocks = sorted(int(k.split("_")[1]) for k in data.files if k.startswith("block_"))
    feats  = {b: data[f"block_{b}"] for b in blocks}      # (N, D) per block
    return feats, labels, blocks


def _cka_matrix(feats_a: dict, feats_b: dict, blocks: list) -> np.ndarray:
    """Pairwise linear-CKA matrix between two per-block feature dicts.

    feats_a[b] : (n_a, D)
    feats_b[b] : (n_b, D)
    n_a may differ from n_b (CKA is sample-aligned via the kernel structure).
    Returns (len(blocks), len(blocks)).
    """
    M = np.zeros((len(blocks), len(blocks)))
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            M[i, j] = linear_cka(feats_a[bi], feats_b[bj])
    return M


def main() -> None:
    apply_pub_style()
    feats, labels, blocks = _load_features()
    n_blocks = len(blocks)
    print(f"[q1_cka] computing CKA over {n_blocks} blocks ({blocks})")

    xi = np.where(labels == "xray")[0]
    mi = np.where(labels == "mri")[0]
    feats_x = {b: feats[b][xi] for b in blocks}
    feats_m = {b: feats[b][mi] for b in blocks}

    print("[q1_cka] X-ray self-CKA...")
    cka_xx = _cka_matrix(feats_x, feats_x, blocks)
    print("[q1_cka] MRI self-CKA...")
    cka_mm = _cka_matrix(feats_m, feats_m, blocks)
    print("[q1_cka] cross-modality CKA...")
    cka_xm = _cka_matrix(feats_x, feats_m, blocks)

    diag_xm = np.diag(cka_xm)
    print(f"[q1_cka] cross-modality diagonal (within-block):")
    for b, v in zip(blocks, diag_xm):
        print(f"           block {b:2d}  CKA(X-ray, MRI) = {v:.3f}")

    # =========================== Fig 8 ===========================
    fig = plt.figure(figsize=(7.0, 4.6))
    gs  = GridSpec(2, 3, height_ratios=[3.2, 1.4],
                   hspace=0.45, wspace=0.30)

    panels = [
        (cka_xx, "(a) X-ray self-CKA",          "viridis"),
        (cka_mm, "(b) MRI self-CKA",            "viridis"),
        (cka_xm, "(c) Cross-modality CKA",      "magma"),
    ]
    ims = []
    for k, (M, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks(range(n_blocks)); ax.set_yticks(range(n_blocks))
        ax.set_xticklabels(blocks, fontsize=7); ax.set_yticklabels(blocks, fontsize=7)
        ax.set_xlabel("block $j$", fontsize=9)
        if k == 0: ax.set_ylabel("block $i$", fontsize=9)
        ims.append(im)
    cb = fig.colorbar(ims[-1], ax=fig.axes, shrink=0.62, pad=0.02, fraction=0.025)
    cb.set_label("linear CKA", fontsize=9)

    # bottom row — within-block cross-modality trace
    ax = fig.add_subplot(gs[1, :])
    ax.plot(blocks, diag_xm, "-o", color=PALETTE["accent"], markersize=4,
            label="CKA(X-ray$_b$, MRI$_b$)")
    ax.set_xlabel("Encoder block $b$")
    ax.set_ylabel("CKA")
    ax.set_xticks(blocks)
    ax.set_ylim(max(0, diag_xm.min() - 0.05), min(1.0, diag_xm.max() + 0.05))
    ax.grid(True, alpha=0.3)
    ax.set_title("(d) Within-block cross-modality similarity (diagonal of panel c)",
                 loc="left", fontsize=10)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Fig. 8 — Layer-wise representational similarity (linear CKA). "
        "Panels (a), (b) show how each modality's representation evolves block-to-block; "
        "panel (c) shows X-ray vs MRI alignment at every (block$_i$, block$_j$) pair; "
        "panel (d) is the diagonal of (c).",
        y=1.04, fontsize=9.5,
    )
    save_fig(fig, "fig8_layerwise_cka")
    plt.close(fig)

    # ---- dump matrices for downstream use ----
    np.savez(config.OUT_DIR / "q8_cka.npz",
             blocks=np.array(blocks),
             cka_xx=cka_xx, cka_mm=cka_mm, cka_xm=cka_xm,
             cross_diag=diag_xm)
    print(f"[q1_cka] done -> figures/fig8_layerwise_cka.pdf, "
          f"outputs/q8_cka.npz")


if __name__ == "__main__":
    main()
