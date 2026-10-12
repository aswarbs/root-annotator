"""Y/X tiling for the ConvLSTM tracker (copied from rootseg-lstm/eval.py): a whole slice doesn't
fit through the network, so each slice is cut into patch x patch tiles that are tracked
independently (closed loop, no shared state across tiles) and stitched back together.
"""


def grid_starts(total, patch):
    """Tile start positions along one axis, spaced by `patch`; the last one is snapped flush
    to the edge so the final tile is never a thin sliver."""
    if total <= patch:
        return [0]
    return list(range(0, total - patch, patch)) + [total - patch]


def build_tiles(H, W, patch):
    """(y0, x0, owned_y0, owned_y1, owned_x0, owned_x1) per tile -- the owned box is what this
    tile writes into the stitched output, so an overlap at a flush seam is counted once."""
    ys, xs = grid_starts(H, patch), grid_starts(W, patch)
    tiles = []
    for iy, y0 in enumerate(ys):
        oy0 = y0 if iy == 0 else ys[iy - 1] + patch
        oy1 = y0 + patch if iy < len(ys) - 1 else H
        for ix, x0 in enumerate(xs):
            ox0 = x0 if ix == 0 else xs[ix - 1] + patch
            ox1 = x0 + patch if ix < len(xs) - 1 else W
            tiles.append((y0, x0, oy0, oy1, ox0, ox1))
    return tiles
