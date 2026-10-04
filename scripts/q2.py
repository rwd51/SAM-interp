"""Q2 — does SAM separate X-ray from MRI in its last 4 blocks?

Outputs:
    figures/fig4_separation_metrics_q2.{pdf,png}
    figures/fig5_pca_umap_q2.{pdf,png}
    outputs/q2_metrics.csv
    outputs/q2_pixel_baseline.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_all
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


PIXEL_HW = 32     # raw-pixel baseline resolution (32x32 grey = 1024-d)


def _pixel_features(labels: np.ndarray) -> np.ndarray:
    """Downsampled raw pixels, same order as q1_features.npz (X-rays then MRIs).

    If a linear probe on these already separates the modalities, SAM's
    separation says nothing about what SAM learned — the classes differ in
    low-level image statistics alone.
    """
    images = load_all()
    img_labels = np.array([lab for _, lab in images])
    if not np.array_equal(img_labels, labels):
        raise RuntimeError("image cache and q1_features.npz disagree on labels; "
                           "rerun `python -m scripts.q1`.")
    return np.stack([
        np.asarray(img.convert("L").resize((PIXEL_HW, PIXEL_HW)), np.float32).ravel() / 255.0
        for img, _ in images
    ])


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

    # ---- raw-pixel control ----
    pix = separation_metrics(_pixel_features(labels), y)
    pd.DataFrame([pix]).to_csv(config.OUT_DIR / "q2_pixel_baseline.csv", index=False)
    print(f"[q2] raw-pixel baseline ({PIXEL_HW}x{PIXEL_HW}): " +
          "  ".join(f"{k}={v:.3f}" for k, v in pix.items()))

    # ================== Fig 4 — metrics bar ==================
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2),
                             gridspec_kw=dict(wspace=0.50))
    cols   = ["linear_probe", "knn5", "silhouette", "fisher"]
    titles = ["Linear probe (5-fold)", "5-NN (CV acc)", "Silhouette", "Fisher ratio"]
    colors = [PALETTE["accent"], PALETTE["accent"], PALETTE["warn"], PALETTE["warn"]]

    for ax, col, t, c in zip(axes, cols, titles, colors):
        vals = met_df[col].values
        bars = ax.bar(range(len(last4)), vals, color=c, alpha=0.88, width=0.62,
                      edgecolor="black", linewidth=0.7)
        ax.set_xticks(range(len(last4)))
        ax.set_xticklabels([f"B{b}" for b in last4], fontsize=11)
        ax.set_title(t, loc="left", fontsize=12)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10)
        ax.axhline(pix[col], color="black", ls="--", lw=1.0,
                   label="raw pixels" if col == cols[0] else None)
        ax.margins(y=0.22)
        ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.35)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "Per-layer modality separation across the last four SAM ViT-B blocks.",
        y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig4_separation_metrics_q2"); plt.close(fig)

    # ================== Fig 5 — PCA + UMAP ==================
    try:
        import umap
        has_umap = True
    except Exception:
        has_umap = False

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6),
                             gridspec_kw=dict(wspace=0.20, hspace=0.45))
    for col, b in enumerate(last4):
        X = StandardScaler().fit_transform(feats[b])
        Xp = PCA(n_components=2, random_state=config.SEED).fit_transform(X)

        ax = axes[0, col]
        ax.scatter(Xp[y == 0, 0], Xp[y == 0, 1], c=PALETTE["xray"], s=42,
                   edgecolors="white", linewidths=0.6, label="X-ray")
        ax.scatter(Xp[y == 1, 0], Xp[y == 1, 1], c=PALETTE["mri"], s=42,
                   marker="^", edgecolors="white", linewidths=0.6, label="MRI")
        ax.set_title(f"Block {b}  (PCA)", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("PCA-2", fontsize=12); ax.legend(fontsize=11, loc="best")

        ax = axes[1, col]
        if has_umap and len(X) >= 6:
            Xu = umap.UMAP(n_neighbors=min(8, len(X) - 1), min_dist=0.3,
                           random_state=config.SEED).fit_transform(X)
            ax.scatter(Xu[y == 0, 0], Xu[y == 0, 1], c=PALETTE["xray"], s=42,
                       edgecolors="white", linewidths=0.6)
            ax.scatter(Xu[y == 1, 0], Xu[y == 1, 1], c=PALETTE["mri"], s=42,
                       marker="^", edgecolors="white", linewidths=0.6)
            ax.set_title(f"Block {b}  (UMAP)", fontsize=12)
        else:
            ax.text(0.5, 0.5, "UMAP unavailable", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("UMAP-2", fontsize=12)
    fig.suptitle(
        "Low-dimensional projection of mean-pooled patch embeddings.",
        y=1.01, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig5_pca_umap_q2"); plt.close(fig)

    # ---- choose clearest layer ----
    # Average of min-max-normalised metrics, skipping any metric that is flat
    # across blocks (e.g. linear probe saturated at 1.0). A flat metric carries
    # no ranking information, and with a product it would zero every score and
    # make argmax silently return the first block.
    winner = pick_clearest_block(met_df, last4)
    print(f"[q2] clearest-separation block: {winner}")


def pick_clearest_block(met_df: pd.DataFrame, blocks,
                        cols=("silhouette", "fisher", "linear_probe")) -> int:
    normed = []
    for c in cols:
        v = met_df[c].to_numpy(float)
        span = v.max() - v.min()
        if span < 1e-6:
            print(f"[q2]   {c} is flat across blocks ({v.max():.3f}); ignored for ranking")
            continue
        normed.append((v - v.min()) / span)
    if not normed:
        raise RuntimeError("all separation metrics are flat across blocks; "
                           "no block separates more clearly than another")
    return blocks[int(np.argmax(np.mean(normed, axis=0)))]

if __name__ == "__main__":
    main()
