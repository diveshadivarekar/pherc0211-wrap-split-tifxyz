"""Read and write the Vesuvius Challenge ``tifxyz`` surface format.

A tifxyz surface is a *directory* holding a regular 2D grid of 3D points: ``x.tif``, ``y.tif`` and
``z.tif`` (one float32 image each, same size) plus ``meta.json``. The grid is indexed ``[row, col]``
and each cell holds the ``(x, y, z)`` position, in level-0 voxels of the source volume, of one
vertex. Vertex spacing is ``1 / scale`` voxels (``scale`` is *cells per voxel*, e.g. 0.05 for a mesh
sampled every 20 voxels - never the step itself). Invalid vertices are written as ``(-1, -1, -1)``
and the loaders drop every vertex with ``z <= 0``, so valid points must have ``z > 0``. See the
format note in the villa repository (``lasagna/tifxyz_format.md``); this module follows it and
needs only numpy and tifffile, so it also runs where the ``vesuvius`` package (OpenCV, ...) does not.

This project's M3 stage hands over a ``coord_map``: ``(rows, cols, 3)`` ``(z, y, x)`` coordinates
with NaN outside the page. :func:`from_coord_map` turns that into a :class:`Surface`.
"""

from __future__ import annotations

import json
import uuid as uuid_lib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SENTINEL = -1.0
REQUIRED_FILES = ("x.tif", "y.tif", "z.tif", "meta.json")


@dataclass
class Surface:
    """A tifxyz grid. ``xyz`` is ``(rows, cols, 3)`` float32 in ``(x, y, z)`` order, NaN where invalid."""

    xyz: np.ndarray
    scale: tuple[float, float]  # (sx, sy): grid cells per voxel along columns / rows
    uuid: str = field(default_factory=lambda: uuid_lib.uuid4().hex)
    meta: dict = field(default_factory=dict)  # extra meta.json keys, preserved on write

    @property
    def shape(self) -> tuple[int, int]:
        return self.xyz.shape[:2]

    @property
    def valid(self) -> np.ndarray:
        """Vertices that a tifxyz loader would keep (finite and ``z > 0``)."""
        return np.isfinite(self.xyz).all(axis=-1) & (self.xyz[..., 2] > 0)

    @property
    def spacing(self) -> tuple[float, float]:
        """Nominal vertex spacing in voxels (columns, rows) - the reciprocal of ``scale``."""
        return 1.0 / self.scale[0], 1.0 / self.scale[1]

    def bbox(self) -> list[list[float]] | None:
        pts = self.xyz[self.valid]
        if not len(pts):
            return None
        return [pts.min(axis=0).astype(float).tolist(), pts.max(axis=0).astype(float).tolist()]

    def area_vx2(self) -> float:
        """Surface area in voxel^2: two triangles per quad whose four corners are all valid."""
        p = self.xyz.astype(np.float64)
        v = self.valid
        quad = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
        a, b, c, d = p[:-1, :-1], p[1:, :-1], p[:-1, 1:], p[1:, 1:]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1) + 0.5 * np.linalg.norm(np.cross(b - d, c - d), axis=-1)
        return float(area[quad].sum())


def _edge_lengths(xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3D lengths of the horizontal (along columns) and vertical (along rows) grid edges between valid vertices."""
    p = xyz.astype(np.float64)
    h = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)[valid[:, 1:] & valid[:, :-1]]
    v = np.linalg.norm(p[1:] - p[:-1], axis=-1)[valid[1:] & valid[:-1]]
    return h, v


def measure_scale(xyz: np.ndarray) -> tuple[float, float]:
    """``(sx, sy)`` cells-per-voxel measured from the grid's median edge lengths (not assumed)."""
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
    h, v = _edge_lengths(xyz, valid)
    if not len(h) or not len(v):
        raise ValueError("cannot measure the scale: the grid has no pair of adjacent valid vertices in one of its two directions")
    return float(1.0 / np.median(h)), float(1.0 / np.median(v))


def from_coord_map(
    coord_map: np.ndarray,
    offset_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stride: int = 1,
    scale: tuple[float, float] | None = None,
    uuid: str | None = None,
    meta: dict | None = None,
) -> Surface:
    """Convert an M3 ``coord_map`` (rows, cols, 3 in ``(z, y, x)``, NaN outside the page) to a :class:`Surface`.

    ``offset_zyx`` is added to every coordinate: pass the origin of the tile/crop the page was cut
    from so that the surface is expressed in level-0 voxels of the *whole* scroll volume, which is
    what VC3D and the other tools expect. ``stride`` keeps every n-th row and column (a 1-voxel grid
    is needlessly dense for viewing and tracing). ``scale`` is measured from the grid unless given.
    """
    cm = np.asarray(coord_map, dtype=np.float64)
    if cm.ndim != 3 or cm.shape[-1] != 3:
        raise ValueError(f"coord_map must be (rows, cols, 3), got {cm.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cm = cm[::stride, ::stride] + np.asarray(offset_zyx, dtype=np.float64)
    xyz = cm[..., ::-1].astype(np.float32)  # (z, y, x) -> (x, y, z)
    finite = np.isfinite(xyz).all(axis=-1)
    if finite.any() and (xyz[finite][:, 2] <= 0).any():
        raise ValueError(
            "some valid vertices have z <= 0, which tifxyz loaders silently drop; pass offset_zyx "
            "(the tile's origin in the full volume) so coordinates are absolute scroll voxels"
        )
    xyz[~finite] = np.nan
    s = tuple(scale) if scale is not None else measure_scale(xyz)
    return Surface(xyz=xyz, scale=(float(s[0]), float(s[1])), uuid=uuid or uuid_lib.uuid4().hex, meta=dict(meta or {}))


def to_coord_map(surface: Surface) -> np.ndarray:
    """Inverse of :func:`from_coord_map` (without the offset): ``(rows, cols, 3)`` ``(z, y, x)`` with NaN invalid."""
    cm = surface.xyz[..., ::-1].astype(np.float32).copy()
    cm[~surface.valid] = np.nan
    return cm


def write_tifxyz(
    path: str | Path,
    surface: Surface,
    voxel_size_um: float | None = None,
    overwrite: bool = False,
    write_mask: bool = True,
    compression: str | None = "zlib",
) -> Path:
    """Write ``surface`` as a tifxyz directory and return its path.

    Invalid vertices are written as ``(-1, -1, -1)``. ``meta.json`` gets ``format``, ``type``,
    ``uuid``, ``scale`` (``[sx, sy]``), ``bbox``, ``area_vx2`` (and ``area_cm2`` when
    ``voxel_size_um`` is known) plus any extra keys held in ``surface.meta``.
    """
    import tifffile

    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.mkdir(parents=True, exist_ok=True)

    valid = surface.valid
    xyz = np.where(valid[..., None], surface.xyz, np.float32(SENTINEL)).astype(np.float32)
    for i, name in enumerate(("x", "y", "z")):
        tifffile.imwrite(path / f"{name}.tif", np.ascontiguousarray(xyz[..., i]), photometric="minisblack", compression=compression)
    if write_mask:
        tifffile.imwrite(path / "mask.tif", np.where(valid, 255, 0).astype(np.uint8), photometric="minisblack", compression=compression)

    meta = dict(surface.meta)
    meta.update({"format": "tifxyz", "type": meta.get("type", "seg"), "uuid": surface.uuid, "scale": [round(float(s), 9) for s in surface.scale]})
    box = surface.bbox()
    if box is not None:
        meta["bbox"] = box
    area = surface.area_vx2()
    meta["area_vx2"] = area
    if voxel_size_um:
        meta["voxelsize"] = float(voxel_size_um)
        meta["area_cm2"] = area * (voxel_size_um * 1e-4) ** 2
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return path


def read_tifxyz(path: str | Path) -> Surface:
    """Read a tifxyz directory the way the format note prescribes (any numeric TIFF dtype, ``z <= 0`` and ``mask.tif`` honoured)."""
    import tifffile

    path = Path(path)
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a tifxyz directory: missing {', '.join(missing)}")
    meta = json.loads((path / "meta.json").read_text())
    if "scale" not in meta:
        raise ValueError(f"{path / 'meta.json'} has no 'scale'")
    sx, sy = (float(v) for v in meta["scale"])
    chans = [np.asarray(tifffile.imread(path / f"{n}.tif"), dtype=np.float64) for n in "xyz"]
    if any(c.ndim != 2 or c.shape != chans[0].shape for c in chans):
        raise ValueError(f"{path}: x.tif, y.tif and z.tif must be 2D and the same size, got {[c.shape for c in chans]}")
    xyz = np.stack(chans, axis=-1)
    invalid = ~np.isfinite(xyz).all(axis=-1) | (xyz[..., 2] <= 0)
    if (path / "mask.tif").exists():
        mask = np.asarray(tifffile.imread(path / "mask.tif"))
        if mask.ndim == 3:
            mask = mask[..., 0] if mask.shape[-1] <= 4 else mask[0]
        h, w = invalid.shape
        if mask.shape[0] % h == 0 and mask.shape[1] % w == 0:  # an integer multiple of the grid is honoured; anything else is ignored
            fy, fx = mask.shape[0] // h, mask.shape[1] // w
            bad = (mask < 255).reshape(h, fy, w, fx).any(axis=(1, 3))
            invalid |= bad
    xyz[invalid] = np.nan
    extra = {k: v for k, v in meta.items() if k not in ("uuid", "scale")}
    return Surface(xyz=xyz.astype(np.float32), scale=(sx, sy), uuid=str(meta.get("uuid", path.name)), meta=extra)


def check_contract(path: str | Path) -> list[dict]:
    """Machine-checkable problems with a tifxyz directory: ``[{"level": "error"|"warning", "code", "message"}]``.

    Beyond file/shape checks this flags the classic ``scale`` mistakes - the step written instead
    of its reciprocal, or a scale that disagrees with the grid's real vertex spacing.
    """
    path = Path(path)
    issues: list[dict] = []

    def add(level: str, code: str, message: str) -> None:
        issues.append({"level": level, "code": code, "message": message})

    for f in REQUIRED_FILES:
        if not (path / f).exists():
            add("error", "missing-file", f"{f} is missing")
    if issues:
        return issues
    try:
        meta = json.loads((path / "meta.json").read_text())
    except json.JSONDecodeError as e:
        return [{"level": "error", "code": "bad-json", "message": f"meta.json is not valid JSON: {e}"}]
    if meta.get("format") != "tifxyz":
        add("error", "format", f"meta.json 'format' must be 'tifxyz', got {meta.get('format')!r}")
    scale = meta.get("scale")
    if not (isinstance(scale, list) and len(scale) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in scale)):
        add("error", "scale", f"meta.json 'scale' must be [sx, sy] of positive numbers, got {scale!r}")
        return issues
    if any(v > 2.0 for v in scale):  # a 1-voxel grid is scale 1; a step of 20 written by mistake is 20
        add("warning", "scale-is-step", f"scale {scale} is above 2 cells/voxel: scale is the reciprocal of the vertex spacing, not the step")
    try:
        surf = read_tifxyz(path)
    except Exception as e:  # noqa: BLE001 - report whatever the reader tripped on
        add("error", "unreadable", str(e))
        return issues
    if not surf.valid.any():
        add("error", "empty", "no valid vertices")
        return issues
    try:
        mx, my = measure_scale(surf.xyz)
    except ValueError as e:
        add("warning", "scale-unmeasurable", str(e))
    else:
        for axis, declared, measured in (("columns", scale[0], mx), ("rows", scale[1], my)):
            ratio = declared / measured
            if not 0.75 <= ratio <= 1.33:
                add("warning", "scale-mismatch", f"declared scale along {axis} is {declared:.4g} but the grid's median edge length implies {measured:.4g} (x{ratio:.2f})")
    return issues
