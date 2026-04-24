"""Q3 — causal head-ablation sweep + hypothesis tests.

Runs:
    (A) For every (block, head) in the last N blocks, zero the head and measure
        the change in separation of the last-block mean-pooled embedding. This
        gives the causal importance heatmap (Fig. 6).
    (B) Pick the head whose ablation most asymmetrically hurts MRI (the scenario
        the prompt describes) and run two hypothesis tests:
            H1  modality-specialised head -> compare per-modality activation norm
            H2  redundancy asymmetry      -> per-modality CKA + pair-ablation probe
        (Fig. 7)

Outputs:
    figures/fig6_head_importance_q3.{pdf,png}
    figures/fig7_hypothesis_tests_q3.{pdf,png}
    outputs/q3_ablation_sweep.csv

CLI:
    python -m scripts.q3 --sweep-blocks 2        # restrict sweep to last 2 blocks (faster)
    python -m scripts.q3 --sweep-blocks 4        # default: last 4 blocks
"""
from __future__ import annotations

import argparse
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src import config
from src.data import load_all
from src.metrics import linear_cka, separation_metrics
from src.model import (clear_cuda, encode, load_sam,
                       set_ablation, set_ablation_multi, set_store)
from src.viz import PALETTE, apply_pub_style, save_fig


def _pooled(tok):
    return tok.reshape(-1, tok.shape[-1]).mean(0).numpy()


def _encode_pooled_at(target_block: int, images) -> np.ndarray:
    feats = []
    for img, _ in images:
        out = encode(img, block_ids=[target_block])
        feats.append(_pooled(out["tokens"][target_block][0]))
        del out
    clear_cuda()
    return np.stack(feats)


def main(sweep_blocks: int = 4) -> None:
    apply_pub_style()
    _, enc, meta = load_sam()
    n_blocks, n_heads = meta["n_blocks"], meta["n_heads"]
    last4        = list(range(n_blocks - 4, n_blocks))
    sweep_blocks = last4[-sweep_blocks:]
    target       = last4[-1]
    print(f"[q3] sweeping blocks {sweep_blocks}, scoring at block {target}")

    images = load_all()
    labels = np.array([lab for _, lab in images])
    y      = (labels == "mri").astype(int)

    # ---- baseline ----
    set_ablation(None, None)
    base_feats = _encode_pooled_at(target, images)
    base       = separation_metrics(base_feats, y)
    print(f"[q3] baseline: {base}")

    # ==================================================
    # (A) Full (block, head) ablation sweep
    # Per-class score = CV-predicted accuracy on held-out folds, per class.
    # (Training accuracy is near 1.0 for 40x768 features regardless of the
    # ablation, which would make the asymmetry column meaningless.)
    # ==================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    def _per_class_cv_acc(feats: np.ndarray) -> tuple[float, float]:
        Xs = StandardScaler().fit_transform(feats)
        pred = cross_val_predict(LogisticRegression(max_iter=500), Xs, y, cv=cv)
        acc_x = float((pred[y == 0] == 0).mean())
        acc_m = float((pred[y == 1] == 1).mean())
        return acc_x, acc_m

    base_acc_x, base_acc_m = _per_class_cv_acc(base_feats)
    print(f"[q3] baseline per-class CV acc: X-ray={base_acc_x:.3f}, MRI={base_acc_m:.3f}")

    rows = []
    for b, h in tqdm(list(product(sweep_blocks, range(n_heads))),
                     desc="q3/ablating"):
        set_ablation(b, h)
        feats = _encode_pooled_at(target, images)
        m     = separation_metrics(feats, y)
        acc_x, acc_m = _per_class_cv_acc(feats)
        rows.append(dict(
            block=b, head=h,
            sil=m["silhouette"], fisher=m["fisher"], lp=m["linear_probe"],
            acc_xray=acc_x, acc_mri=acc_m,
            delta_acc_xray=base_acc_x - acc_x,
            delta_acc_mri =base_acc_m - acc_m,
            delta_sil=base["silhouette"] - m["silhouette"],
        ))
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(
                config.OUT_DIR / "q3_ablation_sweep_partial.csv", index=False)

    set_ablation(None, None)
    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(config.OUT_DIR / "q3_ablation_sweep.csv", index=False)
    print(abl_df.sort_values("delta_sil", ascending=False).head(6).to_string(index=False))

    # ================== Fig 6 ==================
    heat = abl_df.pivot(index="block", columns="head", values="delta_sil").values
    fig, ax = plt.subplots(figsize=(6.75, 2.0))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(heat).max(), vmax=np.abs(heat).max())
    ax.set_yticks(range(len(sweep_blocks)))
    ax.set_yticklabels([f"block {b}" for b in sweep_blocks])
    ax.set_xticks(range(n_heads)); ax.set_xlabel("Head index")
    ax.set_title(r"Fig. 6 — Causal head-importance ($\Delta$ silhouette when zeroed).",
                 loc="left", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(r"$\Delta$ silhouette", fontsize=9)

    # Asymmetric head = biggest MRI accuracy drop + smallest X-ray accuracy drop.
    # (Matches the Q3 prompt scenario exactly.)
    abl_df["asym"] = abl_df["delta_acc_mri"] - abl_df["delta_acc_xray"]
    top = abl_df.sort_values("asym", ascending=False).iloc[0]
    ax.scatter([top["head"]], [sweep_blocks.index(int(top["block"]))],
               s=80, marker="o", facecolors="none",
               edgecolors="black", linewidths=1.2)
    ax.annotate(f"asym. head\n(B{int(top.block)}, H{int(top.head)})",
                (top["head"], sweep_blocks.index(int(top["block"]))),
                textcoords="offset points", xytext=(8, 8), fontsize=8)
    save_fig(fig, "fig6_head_importance_q3"); plt.close(fig)

    ABL_B = int(top["block"])
    ABL_H = int(top["head"])
    print(f"[q3] asymmetric head: B{ABL_B}.H{ABL_H}")

    # ==================================================
    # (B) Hypothesis tests on the asymmetric head
    # For windowed blocks _head_out has shape (num_windows*B, nH, N_window, d_h);
    # for global blocks (B, nH, N, d_h). Pool by averaging over the batch axis
    # (= windows for windowed blocks) and the spatial axis → always (nH, d_h).
    # ==================================================
    set_ablation(None, None)
    set_store(head_out=True)
    norms_per_img = []      # for H1: mean per-patch norm of each head
    mean_feats    = []      # for H2: mean per-head feature vector
    for img, _ in images:
        _ = encode(img, block_ids=[ABL_B])
        ho = enc.blocks[ABL_B].attn._head_out                # (W_or_B, nH, N, d_h)
        norms_per_img.append(ho.norm(dim=-1).mean(dim=(0, 2)).numpy())   # (nH,)
        mean_feats   .append(ho.mean(dim=(0, 2)).numpy())                # (nH, d_h)
    set_store(head_out=False)
    norms    = np.stack(norms_per_img)                       # (N, nH)
    head_out = np.stack(mean_feats)                          # (N, nH, d_h)

    # --- H2: CKA(head_k, other_head) per modality ------------------------------
    def cka_mat(mask):
        H = head_out[mask]
        M = np.zeros((n_heads, n_heads))
        for i in range(n_heads):
            for j in range(n_heads):
                M[i, j] = linear_cka(H[:, i], H[:, j])
        return M
    CKA_x = cka_mat(labels == "xray")
    CKA_m = cka_mat(labels == "mri")

    # --- H2: pair-ablation probe -----------------------------------------------
    pair_drops = []
    for partner in range(n_heads):
        if partner == ABL_H:
            pair_drops.append(0.0); continue
        set_ablation_multi(ABL_B, [ABL_H, partner])
        feats = _encode_pooled_at(target, images)
        set_ablation(None, None)
        Xs = StandardScaler().fit_transform(feats)
        S  = cosine_similarity(Xs)
        inner = S[np.ix_(np.where(y == 0)[0], np.where(y == 0)[0])]
        cross = S[np.ix_(np.where(y == 0)[0], np.where(y == 1)[0])]
        pair_drops.append(inner.mean() - cross.mean())

    # ================== Fig 7 ==================
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=0.45))
    # (a) H1
    ax = axes[0]
    mean_norms = pd.DataFrame({
        "head":  np.tile(np.arange(n_heads), len(labels)),
        "label": np.repeat(labels, n_heads),
        "n":     norms.reshape(-1),
    }).groupby(["head", "label"])["n"].mean().unstack()
    x_ = np.arange(n_heads)
    ax.bar(x_ - 0.2, mean_norms["xray"], 0.4, color=PALETTE["xray"], label="X-ray",
           edgecolor="black", linewidth=0.5)
    ax.bar(x_ + 0.2, mean_norms["mri"],  0.4, color=PALETTE["mri"],  label="MRI",
           edgecolor="black", linewidth=0.5)
    ax.axvspan(ABL_H - 0.5, ABL_H + 0.5, color="red", alpha=0.12)
    ax.set_xlabel("Head"); ax.set_ylabel(r"$\|$ output $\|_2$")
    ax.set_title(f"(a) H1: per-head norm (B{ABL_B})", loc="left", fontsize=9)
    ax.set_xticks(x_); ax.legend(fontsize=8)

    # (b) H2: CKA to target head
    ax = axes[1]
    ax.bar(x_ - 0.2, CKA_x[ABL_H], 0.4, color=PALETTE["xray"], label="X-ray",
           edgecolor="black", linewidth=0.5)
    ax.bar(x_ + 0.2, CKA_m[ABL_H], 0.4, color=PALETTE["mri"],  label="MRI",
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Other head"); ax.set_ylabel(f"CKA w/ head {ABL_H}")
    ax.set_title("(b) H2: representational overlap", loc="left", fontsize=9)
    ax.set_xticks(x_); ax.legend(fontsize=8)

    # (c) pair-ablation probe
    ax = axes[2]
    ax.bar(x_, pair_drops, 0.7, color=PALETTE["neutral"],
           edgecolor="black", linewidth=0.5)
    ax.axvspan(ABL_H - 0.5, ABL_H + 0.5, color="red", alpha=0.12)
    ax.set_xlabel("Co-ablated partner head")
    ax.set_ylabel("X-ray cluster coherence")
    ax.set_title("(c) H2: pair-ablation probe", loc="left", fontsize=9)
    ax.set_xticks(x_)

    fig.suptitle(
        f"Fig. 7 — Hypothesis tests for asymmetric head (B{ABL_B}, H{ABL_H}).",
        y=1.08, fontsize=9.5)
    save_fig(fig, "fig7_hypothesis_tests_q3"); plt.close(fig)

    # ---- textual verdict ----
    xr = mean_norms.loc[ABL_H, "xray"]; mr = mean_norms.loc[ABL_H, "mri"]
    best_partner = int(np.argmax([pair_drops[i] if i != ABL_H else -1
                                  for i in range(n_heads)]))
    print(f"\n[q3] verdict for B{ABL_B}.H{ABL_H}")
    print(f"     H1 activation-norm ratio MRI/X-ray = {mr / max(xr, 1e-6):.2f}x")
    print(f"     H2 best X-ray redundancy partner  = H{best_partner}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-blocks", type=int, default=4,
                    help="how many trailing blocks to sweep (default 4)")
    args = ap.parse_args()
    main(sweep_blocks=args.sweep_blocks)
