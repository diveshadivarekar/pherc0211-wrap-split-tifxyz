"""M6 — what the viewer shows, kept free of Tkinter so it can be tested and reused headlessly.

A *view* is a list of *panels* (each a 2D array plus bounding boxes to draw on it); the
GUI just lays the panels out on a canvas and draws them. Everything that decides what
a bounding box *is* (segment extents, ink tokens, scroll extent, page extent) lives here.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from src.imaging import apply_hot_colormap, normalize_to_uint8

MAX_SEGMENT_BOXES = 40
MIN_SEGMENT_BOX_PIXELS = 150
MIN_INK_TOKEN_PIXELS = 12
MAX_UPSCALE = 6.0
TITLE_HEIGHT = 24
PANEL_GAP = 12

EXTENT_COLOR = "#2563eb"
INK_OK_COLOR = "#16a34a"
INK_LOW_COLOR = "#d97706"


@dataclass
class Box:
    """A rectangle in *image* pixels (x0, y0 inclusive; x1, y1 exclusive)."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    color: str = EXTENT_COLOR
    info: str = ""


@dataclass
class Panel:
    title: str
    array: np.ndarray  # what is drawn (2D), or the grayscale base when ``rgb`` is given
    kind: str = "gray"  # "gray" | "hot" | "labels" | "rgb"
    boxes: list[Box] = field(default_factory=list)
    rgb: np.ndarray | None = None  # precomputed (H, W, 3) uint8 image, overrides ``array`` for drawing
    values: np.ndarray | None = None  # what the hover read-out reports (defaults to ``array``)

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape[:2]


@dataclass
class View:
    panels: list[Panel]
    caption: str = ""
    notice: str | None = None  # something the user must not miss (e.g. "ink map is a heuristic")
    depth: int | None = None  # number of z slices when the view is scrubbable
    layers: list[str] | None = None  # selectable page layers when the view is per-layer


# --------------------------------------------------------------------------- text helpers


def natural_key(name: str) -> tuple:
    """Sort key so ``layer_2`` comes before ``layer_10``."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def middle_truncate(text: str, max_chars: int) -> str:
    """Shorten a long path/name to ``max_chars`` by eliding the middle (keeps both the start and the file name)."""
    if max_chars < 5 or len(text) <= max_chars:
        return text if len(text) <= max_chars else text[:max_chars]
    keep = max_chars - 1
    head = keep // 2
    return text[:head] + "…" + text[len(text) - (keep - head) :]


# --------------------------------------------------------------------------- colours


def label_color(label_id: int) -> tuple[int, int, int]:
    """Stable, well-separated colour per segment id (golden-ratio hue walk), readable on a light UI."""
    hue = (label_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


# --------------------------------------------------------------------------- bounding boxes


def segment_boxes(label_slice: np.ndarray, min_pixels: int = MIN_SEGMENT_BOX_PIXELS, max_boxes: int = MAX_SEGMENT_BOXES) -> list[Box]:
    """One box per segment visible on this slice (largest ``max_boxes`` with at least ``min_pixels`` pixels)."""
    n = int(label_slice.max()) if label_slice.size else 0
    if n == 0:
        return []
    areas = np.bincount(label_slice.ravel(), minlength=n + 1)
    found = ndi.find_objects(label_slice, max_label=n)
    ids = [i for i in np.argsort(-areas[1:])[:max_boxes] + 1 if areas[i] >= min_pixels and found[i - 1] is not None]
    boxes = []
    for i in ids:
        ys, xs = found[i - 1]
        boxes.append(
            Box(xs.start, ys.start, xs.stop, ys.stop, label=str(i), color=_hex(label_color(i)),
                info=f"segment {i}: {areas[i]:,} px on this slice, box {xs.stop - xs.start}×{ys.stop - ys.start}")
        )
    return boxes


def ink_boxes(ink_map: np.ndarray, threshold: float = 0.5, min_pixels: int = MIN_INK_TOKEN_PIXELS) -> list[Box]:
    """A box per ink blob; amber when the blob's mean confidence is low, green otherwise."""
    from src.m5_ocr_translation.pipeline import extract_ink_tokens

    boxes = []
    for token in extract_ink_tokens(ink_map, threshold=threshold, min_pixels=min_pixels):
        y0, x0, y1, x1 = token["bbox"]
        low = token["flagged"]
        boxes.append(
            Box(x0, y0, x1, y1, color=INK_LOW_COLOR if low else INK_OK_COLOR,
                info=f"ink region, {token['n_pixels']} px, confidence {token['mean_confidence']:.0%}" + (" - LOW, needs review" if low else ""))
        )
    return boxes


def foreground_box(arr: np.ndarray, threshold: float | None = None, label: str = "scroll extent", color: str = EXTENT_COLOR) -> Box | None:
    """Bounding box of the foreground (Otsu threshold unless one is given)."""
    from src.m2_segmentation.segment import estimate_foreground_threshold

    arr = np.nan_to_num(arr.astype(np.float32))
    thr = estimate_foreground_threshold(arr) if threshold is None else threshold
    ys, xs = np.nonzero(arr > thr)
    if len(ys) == 0:
        return None
    return Box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, label=label, color=color,
               info=f"{label}: {xs.max() + 1 - xs.min()}×{ys.max() + 1 - ys.min()} px")


def page_extent_box(coord_map: np.ndarray, label: str = "page extent") -> Box | None:
    """Bounding box of the page pixels that map back to the scroll (the rest is padding)."""
    valid = ~np.isnan(coord_map[..., 0])
    ys, xs = np.nonzero(valid)
    if len(ys) == 0:
        return None
    return Box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, label=label,
               info=f"{label}: {valid.mean():.0%} of the page maps to the scan")


# --------------------------------------------------------------------------- panel -> image


def panel_image(panel: Panel) -> Image.Image:
    """Native-resolution RGB image of a panel (percentile-stretched, so faint structure is visible)."""
    if panel.rgb is not None:
        return Image.fromarray(panel.rgb.astype(np.uint8), mode="RGB")
    if panel.kind == "labels":
        palette = np.array([(0, 0, 0)] + [label_color(i) for i in range(1, int(panel.array.max()) + 1)], dtype=np.uint8)
        return Image.fromarray(palette[panel.array.astype(int)], mode="RGB")
    gray = normalize_to_uint8(panel.array)
    if panel.kind == "hot":
        return Image.fromarray(apply_hot_colormap(gray), mode="RGB")
    return Image.fromarray(gray, mode="L").convert("RGB")


def overlay_labels(gray_slice: np.ndarray, label_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Grayscale slice with segment colours blended on top of the labelled pixels."""
    base = np.repeat(normalize_to_uint8(gray_slice)[..., None], 3, axis=-1).astype(np.float32)
    n = int(label_slice.max())
    if n == 0:
        return base.astype(np.uint8)
    palette = np.array([(0, 0, 0)] + [label_color(i) for i in range(1, n + 1)], dtype=np.float32)
    colour = palette[label_slice.astype(int)]
    mask = (label_slice > 0)[..., None]
    return np.where(mask, (1 - alpha) * base + alpha * colour, base).astype(np.uint8)


# --------------------------------------------------------------------------- layout


@dataclass
class Placed:
    index: int
    x: float
    y: float  # top-left of the image area (below its title)
    w: float
    h: float
    scale: float


def fit_layout(
    shapes: list[tuple[int, int]], width: float, height: float, gap: float = PANEL_GAP, title_h: float = TITLE_HEIGHT, margin: float = 8
) -> list[Placed]:
    """Place panels (given as (rows, cols)) side by side or stacked, whichever lets them show larger,
    with one common scale so panels stay comparable. Never overflows ``width`` x ``height``."""
    n = len(shapes)
    width, height = width - 2 * margin, height - 2 * margin  # keep panels off the canvas edge
    if n == 0 or width < 10 or height < 10:
        return []
    hs, ws = [s[0] for s in shapes], [s[1] for s in shapes]

    s_row = min((width - gap * (n - 1)) / sum(ws), (height - title_h) / max(hs))
    s_col = min(width / max(ws), (height - n * title_h - gap * (n - 1)) / sum(hs))
    horizontal = s_row >= s_col
    scale = max(0.01, min(max(s_row, s_col), MAX_UPSCALE))

    placed: list[Placed] = []
    if horizontal:
        total_w = sum(w * scale for w in ws) + gap * (n - 1)
        x = (width - total_w) / 2
        y0 = (height - (max(hs) * scale + title_h)) / 2 + title_h
        for i in range(n):
            placed.append(Placed(i, x, y0, ws[i] * scale, hs[i] * scale, scale))
            x += ws[i] * scale + gap
    else:
        total_h = sum(h * scale + title_h for h in hs) + gap * (n - 1)
        y = (height - total_h) / 2
        for i in range(n):
            y += title_h
            placed.append(Placed(i, (width - ws[i] * scale) / 2, y, ws[i] * scale, hs[i] * scale, scale))
            y += hs[i] * scale + gap
    return [Placed(p.index, p.x + margin, p.y + margin, p.w, p.h, p.scale) for p in placed]


def hit_test(placed: list[Placed], px: float, py: float) -> tuple[int, float, float] | None:
    """Canvas point -> (panel index, image x, image y), or None if it is outside every panel."""
    for p in placed:
        if p.x <= px < p.x + p.w and p.y <= py < p.y + p.h:
            return p.index, (px - p.x) / p.scale, (py - p.y) / p.scale
    return None


def box_at(panel: Panel, ix: float, iy: float) -> Box | None:
    """The smallest box containing an image point (so nested boxes pick the inner one)."""
    hits = [b for b in panel.boxes if b.x0 <= ix < b.x1 and b.y0 <= iy < b.y1]
    return min(hits, key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0)) if hits else None


def readout(panel: Panel, ix: float, iy: float) -> str:
    h, w = panel.shape
    x, y = int(ix), int(iy)
    if not (0 <= x < w and 0 <= y < h):
        return ""
    src = panel.values if panel.values is not None else panel.array
    value = src[y, x]
    if panel.kind in ("labels",) or (panel.values is not None and np.issubdtype(src.dtype, np.integer)):
        text = f"segment {int(value)}" if value else "background"
    else:
        text = f"value {float(value):.3f}"
    return f"{panel.title}  ·  x {x}, y {y}  ·  {text}"


def compose_png(view: View, size: tuple[int, int] = (1100, 640), show_boxes: bool = True) -> Image.Image:
    """Render a view (panels, titles, boxes) to one image on a light background - used for
    'save view as PNG' and for checking the visuals without a display."""
    img = Image.new("RGB", size, "#ffffff")
    draw = ImageDraw.Draw(img)
    for p in fit_layout([pl.shape for pl in view.panels], *size):
        panel = view.panels[p.index]
        tile = panel_image(panel).resize((max(1, int(p.w)), max(1, int(p.h))), Image.LANCZOS)
        img.paste(tile, (int(p.x), int(p.y)))
        draw.text((p.x, p.y - 18), panel.title, fill="#1f2933")
        draw.rectangle([p.x - 1, p.y - 1, p.x + p.w, p.y + p.h], outline="#dfe3e8")
        if show_boxes:
            for b in panel.boxes:
                draw.rectangle([p.x + b.x0 * p.scale, p.y + b.y0 * p.scale, p.x + b.x1 * p.scale, p.y + b.y1 * p.scale], outline=b.color, width=2)
                if b.label and (b.x1 - b.x0) * p.scale >= 28:
                    draw.text((p.x + b.x0 * p.scale + 3, p.y + b.y0 * p.scale + 2), b.label, fill=b.color)
    return img


# --------------------------------------------------------------------------- loading views


def list_layers(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(f.keys(), key=natural_key)


def _slice(path: str | Path, dataset: str, z: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        ds = f[dataset]
        return ds[min(max(z, 0), ds.shape[0] - 1)]


def volume_depth(path: str | Path, dataset: str = "volume") -> int:
    with h5py.File(path, "r") as f:
        return int(f[dataset].shape[0])


def build_view(kind: str, paths: dict[str, Path | None], z: int = 0, layer: str | None = None, show_extent: bool = True) -> View:
    """Assemble a view from whatever files exist. ``paths`` keys: scan, corrected, labels, flattened, ink."""
    scan, corrected, labels, flattened, ink = (paths.get(k) for k in ("scan", "corrected", "labels", "flattened", "ink"))

    if kind == "scan":
        source = scan or corrected
        if source is None:
            raise ValueError("Open a scan first.")
        panels = [Panel("Raw slice" if scan else "Volume slice", _slice(source, "volume", z))]
        if scan and corrected and Path(corrected).exists():
            panels.append(Panel("Corrected slice", _slice(corrected, "volume", z)))
        for panel in panels:
            box = foreground_box(panel.array)
            if box:
                panel.boxes.append(box)
        return View(panels, f"Slice {z} of the volume", depth=volume_depth(source))

    if kind == "segmentation":
        if corrected is None or labels is None:
            raise ValueError("Run segmentation (M2) or choose a corrected volume and a labels file.")
        vol, lab = _slice(corrected, "volume", z), _slice(labels, "labels", z)
        boxes = segment_boxes(lab)
        panels = [
            Panel("Corrected slice", vol),
            Panel(f"Segments ({len(np.unique(lab[lab > 0]))} on this slice)", vol, kind="rgb", rgb=overlay_labels(vol, lab), values=lab, boxes=boxes),
        ]
        return View(panels, f"Slice {z}; boxes show the {len(boxes)} largest segments on it", depth=volume_depth(corrected))

    if kind in ("flattened", "ink"):
        if flattened is None:
            raise ValueError("Run flattening (M3 / spiral fitting) or choose a flattened-pages file.")
        layers = list_layers(flattened)
        if not layers:
            raise ValueError("The flattened-pages file has no pages.")
        layer = layer if layer in layers else layers[0]
        with h5py.File(flattened, "r") as f:
            grp = f[layer]
            page, coord_map = grp["page"][...], grp["coord_map"][...]
            distortion = grp.attrs.get("distortion_after")
            method = grp.attrs.get("method", "")
        extent = page_extent_box(coord_map) if show_extent else None
        page_panel = Panel(f"{layer} - flattened page", page, boxes=[extent] if extent else [])
        meta = f"{layer}: page {page.shape[1]}×{page.shape[0]}" + (f", {method}" if method else "") + (f", distortion {float(distortion):.3f}" if distortion is not None else "")
        if kind == "flattened":
            return View([page_panel], meta, layers=layers)

        if ink is None:
            raise ValueError("Run ink detection (M4) or choose an ink-map file.")
        with h5py.File(ink, "r") as f:
            if layer not in f:
                raise ValueError(f"No ink map for {layer} in that file.")
            ink_map = f[layer]["ink_probability"][...]
            validated = bool(f.attrs.get("validated", True))
            custom_notice = f.attrs.get("notice")
            if "page" in f[layer]:  # a refined page written by the model run: show exactly what it saw
                page_panel = Panel(f"{layer} - page the model saw", f[layer]["page"][...], boxes=page_panel.boxes)
                refined = f[layer]["coord_map"][...]
                ext = page_extent_box(refined) if show_extent else None
                page_panel.boxes = [ext] if ext else []
        boxes = ink_boxes(ink_map)
        ink_panel = Panel(f"Ink confidence ({len(boxes)} regions)", ink_map, kind="hot", boxes=boxes)
        notice = str(custom_notice) if custom_notice else None if validated else "This ink map is an unvalidated density heuristic (no ink ground truth for the scan) - boxes may be noise, inclusions or damage, not text."
        return View([page_panel, ink_panel], meta, notice=notice, layers=layers)

    raise ValueError(f"Unknown view {kind!r}")
