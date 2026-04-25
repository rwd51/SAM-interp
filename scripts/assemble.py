"""Compose all sub-figures into one teaser + print LaTeX-ready tables."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from PIL import Image

from src import config
from src.viz import apply_pub_style, save_fig


def main() -> None:
    apply_pub_style()
    keys = [
        "fig2_layerwise_q1",
        "fig3_attention_maps_q1",
        "fig4_separation_metrics_q2",
        "fig5_pca_umap_q2",
        "fig6_head_importance_q3",
        "fig7_hypothesis_tests_q3",
    ]
    labels = [
        "(A) Q1 — layer-wise representation drift",
        "(B) Q1 — attention heat-maps across depth",
        "(C) Q2 — per-layer modality separation",
        "(D) Q2 — PCA / UMAP of the last four blocks",
        "(E) Q3 — causal head-importance heat-map",
        "(F) Q3 — hypothesis tests on the asymmetric head",
    ]

    images = {k: Image.open(config.FIG_DIR / f"{k}.png") for k in keys}
    fig = plt.figure(figsize=(7.0, 8.2))
    gs  = GridSpec(6, 1, hspace=0.35,
                   height_ratios=[2.2, 3.0, 1.9, 3.4, 2.0, 2.4])
    for i, (lab, k) in enumerate(zip(labels, keys)):
        ax = fig.add_subplot(gs[i]); ax.axis("off")
        ax.imshow(images[k])
        ax.set_title(lab, loc="left", fontsize=10, fontweight="bold", pad=2)
    fig.suptitle(
        "SAM ViT-B on out-of-distribution medical images — summary of all three analyses",
        y=0.995, fontsize=11)
    save_fig(fig, "fig_teaser_report"); plt.close(fig)

    # ---- Q3 verdict banner ----
    import json
    vpath = config.OUT_DIR / "q3_verdict.json"
    if vpath.exists():
        v = json.loads(vpath.read_text())
        winner = "H3 (distributed)" if v["h3"] else \
                 "H2 (redundancy asymmetry)" if v["h2"] else \
                 "H1 (modality-specialist)" if v["h1"] else "Mixed evidence"
        print("\n" + "=" * 66)
        print(f"Q3 verdict  →  {winner}")
        print(f"Picked head: B{v['abl_block']}.H{v['abl_head']}   "
              f"‖Δsil‖max = {v['max_effect']:.4f}  "
              f"({v['effect_frac']*100:.1f}% of baseline {v['baseline_sil']:.3f})")
        print(f"Norm ratio MRI/X-ray = {v['norm_ratio']:.2f}   "
              f"|ΔCKA| to siblings = {v['cka_delta']:.3f}")
        print("=" * 66)

    # ---- LaTeX tables ----
    met = pd.read_csv(config.OUT_DIR / "q2_metrics.csv")
    abl = pd.read_csv(config.OUT_DIR / "q3_ablation_sweep.csv")

    print(r"% ---------- Table 1 : Q2 separation per last-4 block ----------")
    print(r"\begin{tabular}{lcccc}\toprule")
    print(r"Block & Linear probe & 5-NN CV & Silhouette & Fisher ratio \\\midrule")
    for _, r in met.iterrows():
        print(f"{int(r.block)} & {r.linear_probe:.3f} & {r.knn5:.3f} & "
              f"{r.silhouette:.3f} & {r.fisher:.3f} \\\\")
    print(r"\bottomrule\end{tabular}")

    print("\n" + r"% ---------- Table 2 : Q3 top-3 ablations ----------")
    print(r"\begin{tabular}{ccccc}\toprule")
    print(r"Block & Head & $\Delta$Silh. & Acc. X-ray & Acc. MRI \\\midrule")
    # NB: use r["head"] not r.head — Series.head is a method.
    for _, r in abl.sort_values("delta_sil", ascending=False).head(3).iterrows():
        print(f"{int(r['block'])} & {int(r['head'])} & {r['delta_sil']:.3f} & "
              f"{r['acc_xray']:.2f} & {r['acc_mri']:.2f} \\\\")
    print(r"\bottomrule\end{tabular}")

    print("\n[assemble] artefacts:")
    for p in sorted(config.FIG_DIR.iterdir()):
        print(" ", p.relative_to(config.ROOT))
    for p in sorted(config.OUT_DIR.iterdir()):
        print(" ", p.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
