"""Publication-style matplotlib defaults + colour palette + save_fig helper.

Style follows NeurIPS / CVPR: serif fonts, thin spines, editable-text PDFs,
300 DPI, colour-blind-safe Okabe-Ito palette.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from src.config import FIG_DIR


PALETTE = {
    "xray":    "#0173B2",   # blue
    "mri":     "#DE8F05",   # orange
    "neutral": "#4C4C4C",
    "accent":  "#029E73",   # green
    "warn":    "#D55E00",   # vermillion
}


def apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.labelsize":     11,
        "axes.titlesize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.frameon":     False,
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,      # editable text in PDF
        "ps.fonttype":        42,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "lines.linewidth":    1.4,
        "lines.markersize":   4,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.4,
    })


def save_fig(fig, name: str, fig_dir: Path = FIG_DIR) -> None:
    """Write both vector (PDF, for the report) and raster (PNG, for preview)."""
    pdf = fig_dir / f"{name}.pdf"
    png = fig_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"[viz] saved {pdf.name} + {png.name}")
