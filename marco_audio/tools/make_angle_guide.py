"""
Make the printable A4 angle guide (angle_guide.pdf) used by measure_directions.py.

You clap from the 'YOU' edge of the sheet. Each line is labelled with the angle
the clap should READ when the board's 0° arrow points along that line.

Why the labels go clockwise: turning the board anticlockwise by some angle makes
a fixed sound appear to have turned CLOCKWISE by the same angle, from the
board's point of view.

Run:  python make_angle_guide.py      (writes ../angle_guide.pdf)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "angle_guide.pdf"
W, H = 21.0, 29.7          # A4 in cm
CX, CY = W / 2, 15.0       # centre of the guide
R_LINE = 8.5               # line length (cm) - longer than the board's radius

fig = plt.figure(figsize=(W / 2.54, H / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W), ax.set_ylim(0, H), ax.set_aspect("equal"), ax.axis("off")

# 'Toward you' is straight down the page. A line at paper angle phi measured
# anticlockwise from 'toward you' gets the label (360 - phi) mod 360.
for phi in range(0, 360, 15):
    th = np.deg2rad(-90 + phi)                      # -90 = pointing down the page
    major = phi % 45 == 0
    r0 = 0.6 if major else 6.8
    ax.plot([CX + r0 * np.cos(th), CX + R_LINE * np.cos(th)],
            [CY + r0 * np.sin(th), CY + R_LINE * np.sin(th)],
            color="#0b0b0b" if major else "#898781", lw=1.4 if major else 0.6)
    if major:
        label = (360 - phi) % 360
        ax.text(CX + (R_LINE + 0.9) * np.cos(th), CY + (R_LINE + 0.9) * np.sin(th),
                f"{label}°", ha="center", va="center", fontsize=15, fontweight="bold")

ax.plot([CX - 0.4, CX + 0.4], [CY, CY], color="#0b0b0b", lw=1)
ax.plot([CX, CX], [CY - 0.4, CY + 0.4], color="#0b0b0b", lw=1)

ax.text(CX, 27.6, "Marco mic array - direction test guide", ha="center", fontsize=14, fontweight="bold")
ax.text(CX, 26.6, "Centre the board on the cross, logo up. Turn it so its 0° arrow (the USB edge)\n"
        "points along a line: the clap should then read the angle written on that line.",
        ha="center", va="top", fontsize=9.5, linespacing=1.4)

ax.annotate("", xy=(CX, 3.0), xytext=(CX, 4.8),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#2a78d6"))
ax.text(CX, 2.2, "YOU - clap from this side, ~1 m away, at board height", ha="center",
        fontsize=11, fontweight="bold", color="#2a78d6")

# 10 cm scale bar to check the printer didn't shrink the page
ax.plot([1.5, 11.5], [1.0, 1.0], color="#0b0b0b", lw=1.2)
for x in (1.5, 11.5):
    ax.plot([x, x], [0.8, 1.2], color="#0b0b0b", lw=1.2)
ax.text(12.0, 1.0, "should measure 10 cm (print at 100% / actual size)", va="center", fontsize=8)

fig.savefig(OUT)
print(f"Saved {OUT}")
