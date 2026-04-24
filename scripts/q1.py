"""Q1 — layer-wise behaviour on out-of-distribution medical images.

Outputs:
    figures/fig2_layerwise_q1.{pdf,png}       — attn entropy / within-between cos / token norm
    figures/fig3_attention_maps_q1.{pdf,png}  — mean attention at the 4 global blocks
    outputs/q1_features.npz                   — pooled features per block
"""
from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from src import config
from src.data import load_all
from src.model import clear_cuda, encode, load_sam, set_store
from src.viz import PALETTE, apply_pub_style, save_fig


def _pooled(tok: torch.Tensor) -> torch.Tensor:
    return tok.reshape(-1, tok.shape[-1]).mean(0)


def main() -> None:
    apply_pub_style()
    _, _, meta = load_sam()
    n_blocks   = meta["n_blocks"]
    block_ids  = list(range(n_blocks))
    global_attn = meta["global_attn"]

    images = load_all()
    labels = np.array([lab for _, lab in images])

    pooled_feats = {b: [] for b in block_ids}
    token_norms  = {b: [] for b in block_ids}
    entropy_rows = []

    set_store(attn=False, head_out=False, entropy=True)  # cheap scalar only
    for img, label in tqdm(images, desc="q1/encoding"):
        out = encode(img, block_ids=block_ids)
        for b in block_ids:
            tok = out["tokens"][b][0]
            pooled_feats[b].append(_pooled(tok))
            token_norms [b].append(tok.flatten(0, 1).norm(dim=-1).mean().item())
            ent = out["attn_entropy"].get(b)
            if ent is not None:
                entropy_rows.append({"block": b, "label": label,
                                     "entropy": ent.mean().item()})
        del out; clear_cuda()
    set_store(entropy=False)

    pooled_feats = {b: torch.stack(v).numpy() for b, v in pooled_feats.items()}
    ent_df = pd.DataFrame(entropy_rows)

    # ---- within/between cosine similarity per block ----
    xi = np.where(labels == "xray")[0]
    mi = np.where(labels == "mri")[0]
    within_x, within_m, cross = [], [], []
    for b in block_ids:
        S = cosine_similarity(pooled_feats[b])
        within_x.append(S[np.ix_(xi, xi)][np.triu_indices(len(xi), 1)].mean())
        within_m.append(S[np.ix_(mi, mi)][np.triu_indices(len(mi), 1)].mean())
        cross   .append(S[np.ix_(xi, mi)].mean())
    within_x = np.array(within_x); within_m = np.array(within_m); cross = np.array(cross)

    # ================== Fig 2 ==================
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2),
                             gridspec_kw=dict(wspace=0.35))

    # (a) attention dispersion — only global-attn blocks (key count = 4096 in
    # all four, so entropies are on the same scale).  Windowed-block entropies
    # are bounded by log(196) and would distort the axis.
    ax = axes[0]
    ent_global = ent_df[ent_df["block"].isin(global_attn)]
    for lab, c in [("xray", PALETTE["xray"]), ("mri", PALETTE["mri"])]:
        sub = ent_global[ent_global.label == lab].groupby("block")["entropy"]
        m, s = sub.mean(), sub.std()
        ax.plot(m.index, m.values, "-o", color=c, label=lab.upper(), markersize=3.5)
        ax.fill_between(m.index, m.values - s.values, m.values + s.values,
                        color=c, alpha=0.18, lw=0)
    ax.set_xlabel("Global-attn block"); ax.set_ylabel("Attn. entropy (nats)")
    ax.set_title("(a) Attention dispersion", loc="left")
    ax.set_xticks(global_attn); ax.legend(loc="best")

    # (b) within- vs between-modality cosine similarity
    ax = axes[1]
    ax.plot(block_ids, within_x,  "-o", color=PALETTE["xray"], label="within X-ray", markersize=3.5)
    ax.plot(block_ids, within_m,  "-o", color=PALETTE["mri"],  label="within MRI",   markersize=3.5)
    ax.plot(block_ids, cross,     "--s", color=PALETTE["neutral"], label="X-ray vs MRI",
            markersize=3.5, lw=1.1)
    ax.set_xlabel("Encoder block"); ax.set_ylabel("Cosine similarity")
    ax.set_title("(b) Within- vs between-modality", loc="left")
    ax.set_xticks(block_ids); ax.legend(fontsize=7.5)

    # (c) token-norm growth
    ax = axes[2]
    rows = [{"b": b, "label": labels[i], "n": n}
            for b in block_ids for i, n in enumerate(token_norms[b])]
    norm_df = pd.DataFrame(rows)
    for lab, c in [("xray", PALETTE["xray"]), ("mri", PALETTE["mri"])]:
        sub = norm_df[norm_df.label == lab].groupby("b")["n"]
        m, s = sub.mean(), sub.std()
        ax.plot(m.index, m.values, "-o", color=c, label=lab.upper(), markersize=3.5)
        ax.fill_between(m.index, m.values - s.values, m.values + s.values,
                        color=c, alpha=0.18, lw=0)
    ax.set_xlabel("Encoder block"); ax.set_ylabel(r"$\|\mathbf{z}\|_2$  (mean token)")
    ax.set_title("(c) Token-norm growth", loc="left")
    ax.set_xticks(block_ids); ax.legend()

    fig.suptitle(
        "Fig. 2 — Layer-wise behaviour of SAM ViT-B on OOD medical images. "
        "Modality separation emerges in the last third of the encoder.",
        y=1.08, fontsize=9.5)
    save_fig(fig, "fig2_layerwise_q1"); plt.close(fig)

    # ================== Fig 3 — attention heat-maps ==================
    def _attn_map(img, block_idx, q_xy=(32, 32), head=None):
        set_store(attn=True)
        out = encode(img, block_ids=[block_idx])
        set_store(attn=False)
        a = out["attn"][block_idx][0]
        a = a.mean(0) if head is None else a[head]
        qi = q_xy[0] * meta["patch_hw"] + q_xy[1]
        m  = a[qi].reshape(meta["patch_hw"], meta["patch_hw"]).numpy()
        del out; clear_cuda()
        return m

    sample_x = [img for img, lab in images if lab == "xray"][0]
    sample_m = [img for img, lab in images if lab == "mri" ][0]

    fig, axes = plt.subplots(2, len(global_attn) + 1, figsize=(6.75, 3.0),
                             gridspec_kw=dict(wspace=0.08, hspace=0.12))
    for row, (img, name, c) in enumerate([
        (sample_x, "X-ray", PALETTE["xray"]),
        (sample_m, "MRI",   PALETTE["mri"]),
    ]):
        axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].axis("off")
        axes[row, 0].set_title(name, color=c, fontweight="bold", loc="left", fontsize=10)
        for k, b in enumerate(global_attn):
            m = _attn_map(img, b)
            axes[row, k + 1].imshow(m, cmap="inferno")
            axes[row, k + 1].axis("off")
            if row == 0:
                axes[row, k + 1].set_title(f"block {b}", fontsize=9)

    fig.suptitle(
        "Fig. 3 — Mean attention of the centre query patch at the four global-attention blocks.",
        y=1.05, fontsize=9.5)
    save_fig(fig, "fig3_attention_maps_q1"); plt.close(fig)

    # ---- dump features for Q2 ----
    np.savez(config.OUT_DIR / "q1_features.npz",
             labels=labels,
             **{f"block_{b}": pooled_feats[b] for b in block_ids},
             cross=cross, within_x=within_x, within_m=within_m)
    print(f"[q1] done -> {config.FIG_DIR}, {config.OUT_DIR / 'q1_features.npz'}")


if __name__ == "__main__":
    main()
