"""Shared matplotlib style for Marco audio figures (quiet chrome, thin marks, validated colours).
Colour follows the method in every figure:"""
import matplotlib as mpl

INK = "#0b0b0b"         # primary text
INK2 = "#52514e"        # secondary text
MUTED = "#898781"       # axis labels, notes
GRID = "#e1e0d9"        # hairline gridlines
AXIS = "#c3c2b7"        # baseline / axis
SURFACE = "#ffffff"     # white for print

# Categorical slots in fixed order (validated: CVD-adjacent dE >= 9.1, normal-vision >= 22.9).
# Aqua and yellow are below 3:1 on white, so every line also gets a direct label.
COLOR = {
    "gcc_phat": "#2a78d6",     # slot 1 blue
    "xcorr": "#eb6834",        # slot 2 orange - current method (plain cross-correlation)
    "srp_phat": "#1baf7a",     # slot 3 aqua
    "gcc_robust": "#eda100",   # slot 4 yellow
    "chip": "#e87ba4",         # slot 5 magenta (onboard chip, real data only)
}
NAME = {
    "gcc_phat": "GCC-PHAT",
    "xcorr": "Plain cross-correlation",
    "srp_phat": "SRP-PHAT",
    "gcc_robust": "GCC-PHAT robust",
    "chip": "Chip DOA",
}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })
