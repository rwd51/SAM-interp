"""Q2 — does SAM separate X-ray from MRI in its last 4 blocks?

Outputs:
    figures/fig4_separation_metrics_q2.{pdf,png}
    figures/fig5_pca_umap_q2.{pdf,png}
    outputs/q2_metrics.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config
from src.metrics import separation_metrics
from src.viz import PALETTE, apply_pub_style, save_fig


def _load_features():
    p = config.OUT_DIR / "q1_features.npz"
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Run `python -m scripts.q1` first.")
    data = np.load(p, allow_pickle=True)
    labels = data["labels"]
    blocks = sorted(int(k.split("_")[1]) for k in data.files if k.startswith("block_"))
    feats  = {b: data[f"block_{b}"] for b in blocks}
    return feats, labels, blocks


def main() -> None:
    apply_pub_style()
    feats, labels, blocks = _load_features()
    last4 = blocks[-4:]
    print(f"[q2] analysing blocks {last4}")

    y = (labels == "mri").astype(int)

    # ---- metrics ----
    rows = []
    for b in last4:
        m = separation_metrics(feats[b], y)
        rows.append({"block": b, **m})
    met_df = pd.DataFrame(rows)
    print(met_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    met_df.to_csv(config.OUT_DIR / "q2_metrics.csv", index=False)

    # ================== Fig 4 — metrics bar ==================
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.9),
                             gridspec_kw=dict(wspace=0.45))
    cols   = ["linear_probe", "knn5", "silhouette", "fisher"]
    titles = ["Linear probe 5-fold", "5-NN (CV acc)", "Silhouette", "Fisher ratio"]
    colors = [PALETTE["accent"], PALETTE["accent"], PALETTE["warn"], PALETTE["warn"]]

    for ax, col, t, c in zip(axes, cols, titles, colors):
        vals = met_df[col].values
        bars = ax.bar(range(len(last4)), vals, color=c, alpha=0.85, width=0.65,
                      edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(last4)))
        ax.set_xticklabels([f"B{b}" for b in last4])
        ax.set_title(t, loc="left", fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.margins(y=0.22)
        ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.35)
    fig.suptitle(
        "Fig. 4 — Per-layer modality separation across the last four SAM-ViT-B blocks.",
        y=1.1, fontsize=9.5)
    save_fig(fig, "fig4_separation_metrics_q2"); plt.close(fig)

    # ================== Fig 5 — PCA + UMAP ==================
    try:
        import umap
        has_umap = True
    except Exception:
        has_umap = False

    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.4),
                             gridspec_kw=dict(wspace=0.25, hspace=0.45))
    for col, b in enumerate(last4):
        X = StandardScaler().fit_transform(feats[b])
        Xp = PCA(n_components=2, random_state=config.SEED).fit_transform(X)

        ax = axes[0, col]
        ax.scatter(Xp[y == 0, 0], Xp[y == 0, 1], c=PALETTE["xray"], s=22,
                   edgecolors="white", linewidths=0.5, label="X-ray")
        ax.scatter(Xp[y == 1, 0], Xp[y == 1, 1], c=PALETTE["mri"], s=22, marker="^",
                   edgecolors="white", linewidths=0.5, label="MRI")
        ax.set_title(f"Block {b}  (PCA)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("PCA-2"); ax.legend(fontsize=8)

        ax = axes[1, col]
        if has_umap and len(X) >= 6:
            Xu = umap.UMAP(n_neighbors=min(8, len(X) - 1), min_dist=0.3,
                           random_state=config.SEED).fit_transform(X)
            ax.scatter(Xu[y == 0, 0], Xu[y == 0, 1], c=PALETTE["xray"], s=22,
                       edgecolors="white", linewidths=0.5)
            ax.scatter(Xu[y == 1, 0], Xu[y == 1, 1], c=PALETTE["mri"], s=22, marker="^",
                       edgecolors="white", linewidths=0.5)
            ax.set_title(f"Block {b}  (UMAP)", fontsize=9)
        else:
            ax.text(0.5, 0.5, "UMAP unavailable", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("UMAP-2")
    fig.suptitle(
        "Fig. 5 — Low-dim projection of mean-pooled patch embeddings. "
        "Separation grows with depth.",
        y=1.04, fontsize=9.5)
    save_fig(fig, "fig5_pca_umap_q2"); plt.close(fig)

    # ---- choose clearest layer ----
    def mm(v):
        v = np.array(v, float); lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-9)
    score = (mm(met_df.silhouette) * mm(met_df.fisher) * mm(met_df.linear_probe)) ** (1 / 3)
    winner = last4[int(np.argmax(score))]
    print(f"[q2] clearest-separation block: {winner}")


if __name__ == "__main__":
    main()
