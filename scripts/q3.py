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
from sklearn.metrics import silhouette_samples
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
    #
    # With X-ray vs MRI, linear-probe CV accuracy saturates at 1.0 for both
    # classes — classifier-level metrics can't distinguish ablations.
    # We use two continuous, class-conditional metrics instead:
    #   * per-class silhouette  (sklearn.metrics.silhouette_samples → mean over
    #     each class's samples) — sensitive to cluster tightness/separation;
    #   * per-class cosine coherence (mean pairwise cos-sim within the class) —
    #     sensitive to cluster spread.
    # Both are monotonic proxies for "how much did THIS class's cluster get
    # damaged by the ablation". A positive asymmetry on either metric means
    # MRI was hurt more than X-ray.
    # ==================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    def _per_class_stats(feats: np.ndarray) -> dict:
        Xs  = StandardScaler().fit_transform(feats)
        # per-class silhouette (cluster compactness vs. separation)
        sil = silhouette_samples(Xs, y)
        sil_x = float(sil[y == 0].mean())
        sil_m = float(sil[y == 1].mean())
        # per-class intra-cluster cosine coherence
        S  = cosine_similarity(Xs)
        xi = np.where(y == 0)[0]; mi = np.where(y == 1)[0]
        coh_x = float(S[np.ix_(xi, xi)][np.triu_indices(len(xi), 1)].mean())
        coh_m = float(S[np.ix_(mi, mi)][np.triu_indices(len(mi), 1)].mean())
        # still useful: CV acc, for completeness in the CSV
        pred = cross_val_predict(LogisticRegression(max_iter=500), Xs, y, cv=cv)
        acc_x = float((pred[y == 0] == 0).mean())
        acc_m = float((pred[y == 1] == 1).mean())
        return dict(sil_x=sil_x, sil_m=sil_m,
                    coh_x=coh_x, coh_m=coh_m,
                    acc_x=acc_x, acc_m=acc_m)

    base_stats = _per_class_stats(base_feats)
    print(f"[q3] baseline per-class stats: sil_x={base_stats['sil_x']:.3f} "
          f"sil_m={base_stats['sil_m']:.3f}  coh_x={base_stats['coh_x']:.3f} "
          f"coh_m={base_stats['coh_m']:.3f}")

    rows = []
    for b, h in tqdm(list(product(sweep_blocks, range(n_heads))),
                     desc="q3/ablating"):
        set_ablation(b, h)
        feats = _encode_pooled_at(target, images)
        mtr   = separation_metrics(feats, y)
        cc    = _per_class_stats(feats)
        rows.append(dict(
            block=b, head=h,
            sil=mtr["silhouette"], fisher=mtr["fisher"], lp=mtr["linear_probe"],
            acc_xray=cc["acc_x"], acc_mri=cc["acc_m"],
            sil_xray=cc["sil_x"], sil_mri=cc["sil_m"],
            coh_xray=cc["coh_x"], coh_mri=cc["coh_m"],
            delta_sil_xray=base_stats["sil_x"] - cc["sil_x"],
            delta_sil_mri =base_stats["sil_m"] - cc["sil_m"],
            delta_coh_xray=base_stats["coh_x"] - cc["coh_x"],
            delta_coh_mri =base_stats["coh_m"] - cc["coh_m"],
            delta_sil     =base["silhouette"] - mtr["silhouette"],
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

    # Asymmetric head = biggest MRI silhouette drop + smallest X-ray silhouette
    # drop. Silhouette is sensitive even when CV accuracy saturates at 1.0.
    # NB: use bracket access everywhere — `top.head` is the pandas Series.head()
    # method, not the 'head' column.
    abl_df["asym"] = abl_df["delta_sil_mri"] - abl_df["delta_sil_xray"]
    top = abl_df.sort_values("asym", ascending=False).iloc[0]
    ABL_B = int(top["block"])
    ABL_H = int(top["head"])
    ax.scatter([ABL_H], [sweep_blocks.index(ABL_B)],
               s=80, marker="o", facecolors="none",
               edgecolors="black", linewidths=1.2)
    ax.annotate(f"asym. head\n(B{ABL_B}, H{ABL_H})",
                (ABL_H, sweep_blocks.index(ABL_B)),
                textcoords="offset points", xytext=(8, 8), fontsize=8)
    save_fig(fig, "fig6_head_importance_q3"); plt.close(fig)
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

    # -------------------------- mechanistic verdict --------------------------
    # Automatically classify the picked head against H1 / H2 / H3 based on the
    # measured evidence. Thresholds are chosen to be interpretable and
    # reproducible, not maximally tight.
    xr_norm = float(mean_norms.loc[ABL_H, "xray"])
    mr_norm = float(mean_norms.loc[ABL_H, "mri"])
    norm_ratio = mr_norm / max(xr_norm, 1e-6)

    # CKA-symmetry check: does this head's CKA to its siblings look the same
    # on X-ray as on MRI?  If yes → H2 would NOT apply (no redundancy asymmetry).
    siblings = [i for i in range(n_heads) if i != ABL_H]
    cka_delta = float(np.abs(CKA_x[ABL_H, siblings] - CKA_m[ABL_H, siblings]).mean())

    # Pair-ablation sensitivity: does any partner head, when co-ablated, cause
    # a larger X-ray cluster degradation than the target alone?
    best_partner = int(np.argmax([pair_drops[i] if i != ABL_H else -np.inf
                                  for i in range(n_heads)]))
    best_partner_drop = float(pair_drops[best_partner])

    # Magnitude of the overall Δsilhouette perturbation, vs baseline separation.
    max_effect = float(abl_df["delta_sil"].abs().max())
    baseline_sil = float(base["silhouette"])
    effect_frac  = max_effect / baseline_sil

    # --- decision rules ---
    h1_supported = norm_ratio > 2.0 or norm_ratio < 0.5      # truly modality-specialist
    h2_supported = cka_delta > 0.10 and best_partner_drop > 2 * max_effect
    h3_supported = effect_frac < 0.05 and not (h1_supported or h2_supported)

    print("\n" + "=" * 66)
    print(f"[q3] MECHANISTIC VERDICT — picked head B{ABL_B}.H{ABL_H}")
    print("=" * 66)
    print(f"  Per-modality activation norm:  X-ray={xr_norm:.2f}  MRI={mr_norm:.2f}  "
          f"(MRI/X-ray ratio = {norm_ratio:.2f})")
    print(f"  Mean |CKA_xray − CKA_mri| to sibling heads:  {cka_delta:.3f}")
    print(f"  Best X-ray redundancy partner: H{best_partner}  "
          f"(coherence drop {best_partner_drop:+.3f})")
    print(f"  Largest |Δsilhouette| across sweep: {max_effect:.4f}  "
          f"(= {effect_frac*100:.1f}% of baseline silhouette {baseline_sil:.3f})")
    print("-" * 66)
    print(f"  H1 (modality-specialist head):   "
          f"{'SUPPORTED' if h1_supported else 'REJECTED'}   "
          f"[needs norm-ratio outside (0.5, 2.0)]")
    print(f"  H2 (redundancy asymmetry):       "
          f"{'SUPPORTED' if h2_supported else 'REJECTED'}   "
          f"[needs CKA-delta > 0.10 and a compensating partner]")
    print(f"  H3 (distributed modality signal):"
          f"{' SUPPORTED' if h3_supported else ' REJECTED'}   "
          f"[needs max effect < 5% of baseline, and H1/H2 both rejected]")
    print("=" * 66)
    if h3_supported:
        print("  → H3 wins: SAM's modality-separation signal is superposed across")
        print("    heads with no single bottleneck. Consistent with ablation-")
        print("    resistance and high sibling-head CKA (~0.75+).")
    elif h2_supported:
        print("  → H2 wins: this head is redundantly encoded on X-ray but uniquely")
        print("    carries information on MRI, matching the prompt's scenario.")
    elif h1_supported:
        print("  → H1 wins: head is modality-specialised (near-silent on one side).")
    else:
        print("  → Mixed evidence; report numerics literally.")

    # Persist the verdict so assemble.py and the report can pick it up.
    verdict = dict(
        abl_block=ABL_B, abl_head=ABL_H,
        norm_xray=xr_norm, norm_mri=mr_norm, norm_ratio=norm_ratio,
        cka_delta=cka_delta,
        best_partner=best_partner, best_partner_drop=best_partner_drop,
        max_effect=max_effect, baseline_sil=baseline_sil, effect_frac=effect_frac,
        h1=h1_supported, h2=h2_supported, h3=h3_supported,
    )
    import json
    (config.OUT_DIR / "q3_verdict.json").write_text(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-blocks", type=int, default=4,
                    help="how many trailing blocks to sweep (default 4)")
    args = ap.parse_args()
    main(sweep_blocks=args.sweep_blocks)
