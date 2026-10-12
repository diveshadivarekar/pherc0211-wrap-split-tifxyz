"""Out-of-core volumetric I/O.

Backs FR-1.1 / NFR-2: no module may assume a full scan volume fits in memory,
so every read/write here goes through a chunked, on-disk dataset instead of
an in-memory NumPy array.

HDF5 (via h5py) is the PoC backend since it needs no extra native
dependencies beyond h5py itself. OME-Zarr is the target format referenced in
the architecture doc; :class:`VolumeStore` is deliberately the only place
that knows about the on-disk representation, so a Zarr backend can be added
later without touching preprocessing or phantom-generation code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

DATASET_NAME = "volume"


def base_stem(path: str | Path) -> str:
    """File stem with any trailing ``_corrected`` suffix(es) removed, so every
    phase derives the same scan name (``x_corrected.h5`` -> ``x``). Only the
    *trailing* suffix is stripped — a plain ``str.replace`` also mangled names
    that merely contained the word in the middle."""
    stem = Path(path).stem
    while stem.endswith("_corrected") and len(stem) > len("_corrected"):
        stem = stem[: -len("_corrected")]
    return stem


def available_memory_gb() -> float:
    """Free physical memory in GB (Windows via ctypes, POSIX via sysconf); 8 GB if unknown."""
    try:
        import ctypes
        import os

        if os.name == "nt":
            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullAvailPhys / 1e9
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, AttributeError, ValueError):
        return 8.0


class VolumeStore:
    """A chunked, out-of-core 3D voxel volume backed by an HDF5 file.

    Iterating a store yields ``(slice_index, chunk)`` pairs along axis 0
    (the axis manuscript pages/wraps are stacked along), so callers can
    process a multi-terabyte scan one chunk at a time.
    """

    def __init__(self, path: str | Path, mode: str = "r"):
        self.path = Path(path)
        self._file = h5py.File(self.path, mode)

    @classmethod
    def create(
        cls,
        path: str | Path,
        shape: tuple[int, int, int],
        dtype: np.dtype = np.float32,
        chunk_depth: int = 32,
        attrs: dict | None = None,
    ) -> "VolumeStore":
        """Create a new on-disk volume of the given shape, chunked along axis 0."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = h5py.File(path, "w")
        chunks = (min(chunk_depth, shape[0]), shape[1], shape[2])
        f.create_dataset(DATASET_NAME, shape=shape, dtype=dtype, chunks=chunks)
        if attrs:
            for k, v in attrs.items():
                f[DATASET_NAME].attrs[k] = v
        f.close()
        return cls(path, mode="r+")

    @property
    def dataset(self) -> h5py.Dataset:
        return self._file[DATASET_NAME]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dataset.shape

    @property
    def dtype(self) -> np.dtype:
        return self.dataset.dtype

    @property
    def attrs(self) -> dict:
        return dict(self.dataset.attrs)

    @property
    def chunk_depth(self) -> int:
        chunks = self.dataset.chunks
        return chunks[0] if chunks else self.shape[0]

    def iter_chunks(self, depth: int | None = None) -> Iterator[tuple[slice, np.ndarray]]:
        """Yield (axis-0 slice, chunk array) pairs covering the whole volume."""
        depth = depth or self.chunk_depth
        n = self.shape[0]
        for start in range(0, n, depth):
            stop = min(start + depth, n)
            sl = slice(start, stop)
            yield sl, self.dataset[sl, :, :]

    def write_chunk(self, sl: slice, chunk: np.ndarray) -> None:
        self.dataset[sl, :, :] = chunk

    def read_all(self) -> np.ndarray:
        """Materialise the full volume in memory. Only for small phantoms/tests."""
        return self.dataset[...]

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "VolumeStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
