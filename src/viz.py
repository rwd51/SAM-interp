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
    """Larger fonts + more padding so figures stay legible when scaled into a
    LaTeX report (where multi-panel PDFs at \\textwidth shrink each subplot)."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          12,
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "axes.titlepad":      8,         # more room above subplot titles
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "legend.frameon":     False,
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,      # more breathing room around the figure
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "axes.linewidth":     0.9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.major.width":  0.9,
        "ytick.major.width":  0.9,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "lines.linewidth":    1.7,
        "lines.markersize":   5,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.4,
        "figure.constrained_layout.use": False,    # we use tight_layout / subplots_adjust manually
    })


def save_fig(fig, name: str, fig_dir: Path = FIG_DIR) -> None:
    """Write both vector (PDF, for the report) and raster (PNG, for preview)."""
    pdf = fig_dir / f"{name}.pdf"
    png = fig_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"[viz] saved {pdf.name} + {png.name}")
