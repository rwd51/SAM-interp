"""Bootstrap 95% confidence intervals on Q1 cross-modality CKA + Q2 silhouette
and Fisher metrics.

Reads cached `outputs/q1_features.npz` — no SAM forwards required, no GPU
needed. Produces:

    figures/fig4b_q2_bootstrap.{pdf,png}    silhouette + Fisher with 95% CIs
    figures/fig8b_cka_bootstrap.{pdf,png}   cross-mod CKA diagonal with 95% CIs
    outputs/bootstrap_ci.csv                tidy table of all CIs

Method: 1000 stratified bootstrap resamples with replacement (preserving the
20-X-ray / 20-MRI balance). A tiny 1e-6·σ Gaussian jitter is added on each
resample to break duplicate ties — without it, silhouette is undefined when
two resampled rows are identical.

Standard non-parametric percentile bootstrap (Efron & Tibshirani 1993).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.metrics import fisher_ratio, linear_cka
from src.viz import PALETTE, apply_pub_style, save_fig


N_BOOT = 1000
JITTER = 1e-6
PERCENTILES = (2.5, 97.5)            # 95% CI


def _load():
    p = config.OUT_DIR / "q1_features.npz"
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Run `python -m scripts.q1` first.")
    d = np.load(p, allow_pickle=True)
    labels = d["labels"]
    blocks = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("block_"))
    feats  = {b: d[f"block_{b}"] for b in blocks}
    return feats, labels, blocks


def _stratified_resample(rng, xi, mi):
    """Stratified bootstrap: resample xrays-with-replacement and mris-with-
    replacement separately, then concatenate. Preserves class balance."""
    bx = rng.choice(xi, size=len(xi), replace=True)
    bm = rng.choice(mi, size=len(mi), replace=True)
    return np.concatenate([bx, bm])


def _bootstrap_q2(feats, labels, blocks, n_boot=N_BOOT, seed=config.SEED):
    """Bootstrap silhouette + Fisher on the last 4 blocks."""
    rng = np.random.default_rng(seed)
    y  = (labels == "mri").astype(int)
    xi = np.where(y == 0)[0]; mi = np.where(y == 1)[0]
    last4 = blocks[-4:]

    sil_dist    = {b: [] for b in last4}
    fisher_dist = {b: [] for b in last4}

    for _ in range(n_boot):
        idx = _stratified_resample(rng, xi, mi)
        yb  = y[idx]
        for b in last4:
            Xb = feats[b][idx] + rng.normal(0, JITTER, size=feats[b][idx].shape)
            Xs = StandardScaler().fit_transform(Xb)
            sil_dist[b].append(silhouette_score(Xs, yb))
            fisher_dist[b].append(fisher_ratio(Xs, yb))

    rows = []
    for b in last4:
        sil = np.array(sil_dist[b]); fish = np.array(fisher_dist[b])
        rows.append(dict(
            block=b,
            sil_mean = sil.mean(),  sil_lo = np.percentile(sil, PERCENTILES[0]),
                                    sil_hi = np.percentile(sil, PERCENTILES[1]),
            fish_mean = fish.mean(), fish_lo = np.percentile(fish, PERCENTILES[0]),
                                     fish_hi = np.percentile(fish, PERCENTILES[1]),
        ))
    return pd.DataFrame(rows), sil_dist, fisher_dist


def _bootstrap_cka_diag(feats, labels, blocks, n_boot=N_BOOT, seed=config.SEED + 1):
    """Bootstrap the within-block cross-modality CKA across all 12 blocks."""
    rng = np.random.default_rng(seed)
    xi = np.where(labels == "xray")[0]
    mi = np.where(labels == "mri" )[0]

    cka_dist = {b: [] for b in blocks}
    for _ in range(n_boot):
        bx = rng.choice(xi, size=len(xi), replace=True)
        bm = rng.choice(mi, size=len(mi), replace=True)
        for b in blocks:
            A = feats[b][bx] + rng.normal(0, JITTER, size=feats[b][bx].shape)
            B = feats[b][bm] + rng.normal(0, JITTER, size=feats[b][bm].shape)
            cka_dist[b].append(linear_cka(A, B))

    rows = []
    for b in blocks:
        v = np.array(cka_dist[b])
        rows.append(dict(
            block=b, cka_mean=v.mean(),
            cka_lo=np.percentile(v, PERCENTILES[0]),
            cka_hi=np.percentile(v, PERCENTILES[1]),
        ))
    return pd.DataFrame(rows), cka_dist


def _significant_pair(dist_a, dist_b):
    """Non-parametric two-sided test: fraction of bootstrap iterations in which
    metric_a > metric_b. Returns p ≈ 1−|fraction−0.5|*2."""
    a = np.array(dist_a); b = np.array(dist_b)
    diff = a - b
    p_gt = (diff > 0).mean()
    return p_gt, 2 * min(p_gt, 1 - p_gt)         # two-sided p-value


def main() -> None:
    apply_pub_style()
    feats, labels, blocks = _load()
    print(f"[bootstrap] computing CIs over {N_BOOT} resamples")

    # ---- Q2 ----
    print("[bootstrap] Q2: silhouette + Fisher on last 4 blocks...")
    q2_df, sil_dist, fisher_dist = _bootstrap_q2(feats, labels, blocks)
    print(q2_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # block-pair significance (B8 vs each other in the last 4)
    last4 = blocks[-4:]; b8 = last4[0]
    print("\n[bootstrap] B8 vs other blocks — 2-sided p-values on silhouette:")
    for b in last4[1:]:
        _, p = _significant_pair(sil_dist[b8], sil_dist[b])
        print(f"   B{b8} vs B{b}:  p = {p:.4f}  "
              f"({'significant' if p < 0.05 else 'NOT significant'} at α=0.05)")

    # ---- Q1 cross-CKA diagonal ----
    print("\n[bootstrap] Q1.5: cross-modality CKA diagonal, all 12 blocks...")
    cka_df, cka_dist = _bootstrap_cka_diag(feats, labels, blocks)
    print(cka_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- Fig 4b: Q2 metrics with CIs ----
    # Sized for half-width (subfigure) placement at ~0.49 \textwidth.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0),
                             gridspec_kw=dict(wspace=0.40))
    for ax, col_mean, col_lo, col_hi, ylabel, color in [
        (axes[0], "sil_mean",  "sil_lo",  "sil_hi",  "Silhouette",   PALETTE["warn"]),
        (axes[1], "fish_mean", "fish_lo", "fish_hi", "Fisher ratio", PALETTE["warn"]),
    ]:
        x = np.arange(len(q2_df))
        m = q2_df[col_mean].values
        lo = q2_df[col_lo].values; hi = q2_df[col_hi].values
        yerr = np.array([m - lo, hi - m])
        ax.errorbar(x, m, yerr=yerr, fmt="o-", color=color, capsize=4,
                    markersize=7, linewidth=1.8,
                    ecolor=PALETTE["neutral"], elinewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"B{b}" for b in q2_df["block"]], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12); ax.set_xlabel("Block", fontsize=12)
        ax.grid(axis="y", alpha=0.35)
    fig.suptitle(
        "Q2 separation metrics with 95% bootstrap CIs (n=1000 resamples).",
        y=1.00, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig4b_q2_bootstrap"); plt.close(fig)

    # ---- Fig 8b: cross-CKA diagonal with CIs ----
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(cka_df))
    m  = cka_df["cka_mean"].values
    lo = cka_df["cka_lo"].values; hi = cka_df["cka_hi"].values
    yerr = np.array([m - lo, hi - m])
    ax.errorbar(x, m, yerr=yerr, fmt="o-", color=PALETTE["accent"], capsize=4,
                markersize=7, linewidth=2.0,
                ecolor=PALETTE["neutral"], elinewidth=1.1,
                label="cross-modality CKA, 95% CI")
    peak_b = int(cka_df.iloc[np.argmax(m)]["block"])
    ax.axvline(np.argmax(m), color=PALETTE["warn"], ls="--", lw=1.2,
               label=f"peak: block {peak_b}")
    ax.set_xticks(x)
    ax.set_xticklabels(cka_df["block"], fontsize=11)
    ax.set_xlabel("Encoder block", fontsize=12)
    ax.set_ylabel(r"CKA(X-ray$_b$, MRI$_b$)", fontsize=12)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="lower right", fontsize=11)
    fig.suptitle(
        "Within-block cross-modality CKA, 95% bootstrap CIs.",
        y=1.00, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig8b_cka_bootstrap"); plt.close(fig)

    # ---- save tidy CSV ----
    q2_df["metric_set"]  = "q2"
    cka_df["metric_set"] = "cka"
    out = pd.concat([q2_df, cka_df], ignore_index=True)
    out.to_csv(config.OUT_DIR / "bootstrap_ci.csv", index=False)
    print(f"\n[bootstrap] saved figures/fig4b_q2_bootstrap.* + figures/fig8b_cka_bootstrap.*"
          f" + outputs/bootstrap_ci.csv")


if __name__ == "__main__":
    main()
