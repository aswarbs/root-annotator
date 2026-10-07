"""Generates packaging/icon.ico -- a simple green sprout mark, matching the app's plant theme.
Run once with: .venv312/Scripts/python.exe packaging/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
GREEN = (76, 154, 106, 255)
WHITE = (255, 255, 255, 255)


def make_leaf(w, h):
    leaf = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(leaf)
    d.ellipse([0, 0, w - 1, h - 1], fill=WHITE)
    return leaf


def main():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad = 10
    d.rounded_rectangle([pad, pad, SIZE - pad, SIZE - pad], radius=60, fill=GREEN)

    cx, cy = SIZE // 2, SIZE // 2 + 40

    # stem, curving slightly is overkill for an icon -- a straight line reads fine at small sizes
    d.line([(cx, SIZE - 50), (cx, cy)], fill=WHITE, width=14)

    # two narrow leaves meeting at the top of the stem, flaring up and outward
    leaf_w, leaf_h = 118, 30
    left = make_leaf(leaf_w, leaf_h).rotate(55, expand=True, resample=Image.BICUBIC)
    right = make_leaf(leaf_w, leaf_h).rotate(-55, expand=True, resample=Image.BICUBIC)
    img.paste(left, (cx - left.width + 18, cy - left.height + 18), left)
    img.paste(right, (cx - 18, cy - right.height + 18), right)

    out_path = Path(__file__).resolve().parent / "icon.ico"
    img.save(out_path, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
