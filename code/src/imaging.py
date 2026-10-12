"""Shared preview-image rendering: turn a raw float array into something a
human can actually read clearly.

Used by both the GUI (src/m6_gui/app.py, wrapped into a Tk PhotoImage) and
the headless PNG exporter (scripts/export_previews.py) so both share one
normalisation/colouring/upscaling implementation instead of drifting apart.
No torch/tkinter dependency here — just numpy + Pillow — so it's usable
from a pure CLI context.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# A small "hot" colormap (black -> red -> yellow -> white), interpolated —
# good for confidence/probability heatmaps where "brighter = more certain"
# should read intuitively even to someone who's never seen this codebase.
_HOT_CONTROL_POINTS = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [255, 80, 0],
        [255, 200, 0],
        [255, 255, 255],
    ],
    dtype=np.float64,
)

# Stable, distinguishable palette for categorical segment-id overlays.
LABEL_PALETTE = np.array(
    [
        [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
        [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200],
    ],
    dtype=np.uint8,
)


def normalize_to_uint8(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile contrast stretch to 8-bit. Robust to outlier voxels (a
    handful of extreme values shouldn't wash out the whole image), which a
    plain min/max stretch is not."""
    arr = np.nan_to_num(arr.astype(np.float64))
    lo, hi = np.percentile(arr, low_pct), np.percentile(arr, high_pct)
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)


def apply_hot_colormap(gray_uint8: np.ndarray) -> np.ndarray:
    """Map an 8-bit grayscale array to RGB via the hot colormap (linear
    interpolation between control points, matplotlib-free)."""
    xs = np.linspace(0, 255, len(_HOT_CONTROL_POINTS))
    r = np.interp(gray_uint8.ravel(), xs, _HOT_CONTROL_POINTS[:, 0])
    g = np.interp(gray_uint8.ravel(), xs, _HOT_CONTROL_POINTS[:, 1])
    b = np.interp(gray_uint8.ravel(), xs, _HOT_CONTROL_POINTS[:, 2])
    rgb = np.stack([r, g, b], axis=-1).reshape(*gray_uint8.shape, 3)
    return rgb.astype(np.uint8)


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Colourise an integer label array with a fixed, stable palette (label
    0 = background = black)."""
    idx = labels.astype(int) % len(LABEL_PALETTE)
    return LABEL_PALETTE[idx]


def to_display_image(
    arr: np.ndarray,
    size: tuple[int, int] | None = None,
    cmap: str | None = None,
    is_labels: bool = False,
) -> Image.Image:
    """Convert a raw 2D array into a clear, correctly-scaled PIL image.

    - ``is_labels=True``: integer segment ids -> fixed categorical palette.
    - ``cmap="hot"``: continuous values (e.g. an ink-confidence map) -> hot
      colormap, after a percentile contrast stretch.
    - otherwise: plain grayscale, after a percentile contrast stretch.

    Upscaling (when ``size`` is larger than ``arr``) uses LANCZOS
    resampling — sharper and far less blocky than nearest-neighbour for the
    kind of 2x-8x preview enlargement these panels need; downscaling uses
    the same filter, which is also the right choice there (proper
    anti-aliasing instead of moiré from naive decimation).
    """
    if is_labels:
        rgb = labels_to_rgb(arr)
        img = Image.fromarray(rgb, mode="RGB")
    else:
        gray = normalize_to_uint8(arr)
        if cmap == "hot":
            img = Image.fromarray(apply_hot_colormap(gray), mode="RGB")
        else:
            img = Image.fromarray(gray, mode="L")

    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def save_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    out_path: str,
    size: tuple[int, int] = (480, 480),
    left_cmap: str | None = None,
    right_cmap: str | None = None,
    left_is_labels: bool = False,
    right_is_labels: bool = False,
    gap: int = 12,
) -> None:
    """Render two arrays side by side into one PNG — the standard
    before/after preview shape used throughout the pipeline."""
    left_img = to_display_image(left, size=size, cmap=left_cmap, is_labels=left_is_labels).convert("RGB")
    right_img = to_display_image(right, size=size, cmap=right_cmap, is_labels=right_is_labels).convert("RGB")

    canvas = Image.new("RGB", (size[0] * 2 + gap, size[1]), color=(24, 24, 24))
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (size[0] + gap, 0))
    canvas.save(out_path)
