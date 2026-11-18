"""
Make a printable checkerboard PDF at an exact physical size (for calib_capture.py).

  python make_board.py                         # A3, 8x6 squares of 45 mm  (recommended)
  python make_board.py --paper A4 --square 33  # A4, 33 mm squares (lens views up close only)

The page has a 100 mm ruler line: print at 100% / "actual size" and check that the line is
really 100 mm. Then measure a square and pass it to calib_solve.py --square (in metres).

The board is 8 x 6 SQUARES = 7 x 5 INNER corners, the default --pattern of the tools.
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

PAPER_MM = {"A4": (297, 210), "A3": (420, 297)}   # landscape
DPI = 300


def mm(v):
    return int(round(v / 25.4 * DPI))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", choices=sorted(PAPER_MM), default="A3")
    parser.add_argument("--square", type=float, default=None, help="square size in mm (default: largest that fits)")
    parser.add_argument("--squares", default="8x6", help="squares across x down (default 8x6 = 7x5 inner corners)")
    args = parser.parse_args()
    nx, ny = (int(v) for v in args.squares.lower().split("x"))
    pw, ph = PAPER_MM[args.paper]
    # Printers leave ~5 mm unprinted; 12 mm of white around the board is the quiet zone the
    # detector needs. The ruler and label go in the side margin (landscape has spare width).
    margin = 12
    side = 14          # width reserved on the right for the ruler + label
    fit = min((pw - 2 * margin - side) / nx, (ph - 2 * margin) / ny)
    square = args.square if args.square else float(int(fit))   # whole mm, easier to check
    if square > fit + 1e-9:
        raise SystemExit(f"{square} mm squares do not fit on {args.paper}; max {fit:.1f} mm")

    img = Image.new("L", (mm(pw), mm(ph)), 255)
    draw = ImageDraw.Draw(img)
    bw, bh = nx * square, ny * square
    x0 = (pw - side - bw) / 2
    y0 = (ph - bh) / 2
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                draw.rectangle([mm(x0 + i * square), mm(y0 + j * square),
                                mm(x0 + (i + 1) * square) - 1, mm(y0 + (j + 1) * square) - 1], fill=0)

    # Vertical 100 mm ruler line with ticks in the right margin, and a rotated label.
    rx = pw - margin - 3
    ry0 = (ph - 100) / 2
    draw.line([mm(rx), mm(ry0), mm(rx), mm(ry0 + 100)], fill=0, width=mm(0.5))
    for t in (0, 50, 100):
        draw.line([mm(rx - 3), mm(ry0 + t), mm(rx + 3), mm(ry0 + t)], fill=0, width=mm(0.5))
    try:
        font = ImageFont.truetype("arial.ttf", mm(3.2))
    except OSError:
        font = ImageFont.load_default()
    label = (f"line = 100 mm: print at 100% / actual size  |  {nx}x{ny} squares = "
             f"{nx - 1}x{ny - 1} inner corners, {square:g} mm  ->  --square {square / 1000:g} (measure it)")
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    strip = Image.new("L", (right - left + 4, bottom - top + 4), 255)
    ImageDraw.Draw(strip).text((2 - left, 2 - top), label, fill=0, font=font)
    strip = strip.rotate(90, expand=True)
    img.paste(strip, (mm(rx - 9) - strip.width // 2, (img.height - strip.height) // 2))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_board",
                       f"checkerboard_{args.paper}_{nx}x{ny}_{square:g}mm.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out, "PDF", resolution=DPI)
    print(f"{out}\n  {args.paper} landscape, {nx}x{ny} squares of {square:g} mm "
          f"(board {bw:g} x {bh:g} mm), pattern {nx - 1}x{ny - 1}")


if __name__ == "__main__":
    main()
