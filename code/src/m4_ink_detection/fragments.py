"""Official Vesuvius Challenge ink-detection fragments: loading and caching.

Each fragment folder (as fetched by ``scripts/download_fragments.py``) holds
``surface_volume/NN.tif`` (65 slices sampled across the papyrus surface), ``inklabels.png``
(ground-truth ink, binary) and ``mask.png`` (which pixels are papyrus). Ink on carbonised
papyrus is nearly invisible in any single slice; a model has to read the *stack* of slices
through the surface, which is why everything here is (depth, height, width).

The slices are decoded once into a ``uint16`` ``.npy`` cache so training can memory-map random
crops instead of re-decoding multi-hundred-megabyte TIFFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # these label/mask PNGs exceed Pillow's default decompression-bomb limit

DEFAULT_Z_START = 24  # the ink sits near the middle of the 65-slice surface volume
DEFAULT_N_SLICES = 16
FRAGMENT_NAMES = ("frag1", "frag2", "frag3")


@dataclass
class Fragment:
    name: str
    stack: np.ndarray  # (Z, H, W) uint16, memory-mapped
    labels: np.ndarray  # (H, W) bool, True = ink
    mask: np.ndarray  # (H, W) bool, True = papyrus

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape


def _read_binary_png(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def _read_tif(path: Path) -> np.ndarray:
    import tifffile

    return tifffile.imread(path)


def cache_path(root: Path, z_start: int, n_slices: int) -> Path:
    return root / f"stack_z{z_start}_n{n_slices}.npy"


def build_stack_cache(root: str | Path, z_start: int = DEFAULT_Z_START, n_slices: int = DEFAULT_N_SLICES) -> Path:
    """Decode ``n_slices`` TIFFs starting at slice ``z_start`` into one (Z, H, W) uint16 ``.npy``."""
    root = Path(root)
    out = cache_path(root, z_start, n_slices)
    if out.exists():
        return out
    tifs = sorted((root / "surface_volume").glob("*.tif"))
    if len(tifs) < z_start + n_slices:
        raise FileNotFoundError(f"{root} has {len(tifs)} slices; need {z_start + n_slices}")
    first = _read_tif(tifs[z_start])
    tmp = out.with_suffix(".tmp.npy")
    stack = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(n_slices, *first.shape))
    stack[0] = first
    for i in range(1, n_slices):
        stack[i] = _read_tif(tifs[z_start + i])
    stack.flush()
    del stack
    tmp.replace(out)
    return out


def load_fragment(root: str | Path, z_start: int = DEFAULT_Z_START, n_slices: int = DEFAULT_N_SLICES) -> Fragment:
    root = Path(root)
    stack = np.load(build_stack_cache(root, z_start, n_slices), mmap_mode="r")
    labels, mask = _read_binary_png(root / "inklabels.png"), _read_binary_png(root / "mask.png")
    if labels.shape != stack.shape[1:] or mask.shape != stack.shape[1:]:
        raise ValueError(f"{root.name}: labels {labels.shape}, mask {mask.shape} and slices {stack.shape[1:]} differ in size")
    return Fragment(root.name, stack, labels, mask)


def intensity_stats(fragments: list[Fragment], n_samples: int = 2_000_000, seed: int = 0) -> tuple[float, float]:
    """Mean / std of papyrus pixel values (scaled to 0-1) over the training fragments."""
    rng = np.random.default_rng(seed)
    vals = []
    for frag in fragments:
        ys, xs = np.nonzero(frag.mask[::8, ::8])
        pick = rng.integers(0, len(ys), size=min(n_samples // len(fragments), len(ys)))
        z = rng.integers(0, frag.stack.shape[0], size=len(pick))
        vals.append(frag.stack[z, ys[pick] * 8, xs[pick] * 8].astype(np.float32) / 65535.0)
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std() + 1e-6)
