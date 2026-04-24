"""Download + cache chest X-rays and MRI slices, then save a sanity figure."""
from __future__ import annotations

import matplotlib.pyplot as plt

from src import config, data
from src.viz import PALETTE, apply_pub_style, save_fig


def main() -> None:
    apply_pub_style()
    xrays = data.download_xrays(config.N_XRAY)
    mris  = data.download_mri  (config.N_MRI)

    # Fig 1 — dataset samples
    from PIL import Image
    xs = [Image.open(p) for p in xrays[:4]]
    ms = [Image.open(p) for p in mris [:4]]
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.4))
    for j in range(4):
        axes[0, j].imshow(xs[j], cmap="gray"); axes[0, j].axis("off")
        axes[1, j].imshow(ms[j], cmap="gray"); axes[1, j].axis("off")
    axes[0, 0].set_title("Chest X-ray",   loc="left",
                         color=PALETTE["xray"], fontweight="bold")
    axes[1, 0].set_title("Prostate MRI",  loc="left",
                         color=PALETTE["mri"],  fontweight="bold")
    fig.suptitle("Fig. 1 — Dataset samples (grayscale; tiled to 3 channels for SAM).",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig1_dataset_examples")
    plt.close(fig)


if __name__ == "__main__":
    main()
