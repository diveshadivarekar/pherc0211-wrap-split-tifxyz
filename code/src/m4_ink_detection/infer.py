"""Tiled inference and metrics for the fragment-trained ink model."""

from __future__ import annotations

import numpy as np


def normalize_stack(stack: np.ndarray, mean: float, std: float) -> np.ndarray:
    """uint16 slices -> float32 zero-mean/unit-variance (floats are assumed to be 0-1 already)."""
    x = stack.astype(np.float32)
    if stack.dtype == np.uint16:
        x /= 65535.0
    return (x - mean) / std


def _window(tile: int) -> np.ndarray:
    """Soft 2D weight so overlapping tiles blend without visible seams."""
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    ramp = np.minimum(ramp / (tile / 4), 1.0)
    return np.outer(ramp, ramp)


def predict_stack(
    model,
    stack: np.ndarray,
    mean: float,
    std: float,
    mask: np.ndarray | None = None,
    tile: int = 256,
    stride: int = 192,
    batch_size: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Ink probability for every pixel of a (Z, H, W) surface stack (0 outside ``mask``).

    ``stack`` may be a memory-map: only one tile's worth is read at a time.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    _, h, w = stack.shape
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    ys = [max(0, y) for y in ys]
    xs = [max(0, x) for x in xs]

    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    win = _window(tile)
    jobs = [(y, x) for y in ys for x in xs if mask is None or mask[y : y + tile, x : x + tile].any()]

    def run(batch_jobs):
        tiles = []
        for y, x in batch_jobs:
            crop = np.zeros((stack.shape[0], tile, tile), dtype=np.float32)
            piece = normalize_stack(np.asarray(stack[:, y : y + tile, x : x + tile]), mean, std)
            crop[:, : piece.shape[1], : piece.shape[2]] = piece
            tiles.append(crop)
        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=device != "cpu"):
            probs = torch.sigmoid(model(torch.from_numpy(np.stack(tiles)).to(device))).float().cpu().numpy()
        for (y, x), p in zip(batch_jobs, probs):
            ph, pw = min(tile, h - y), min(tile, w - x)
            acc[y : y + ph, x : x + pw] += p[:ph, :pw] * win[:ph, :pw]
            weight[y : y + ph, x : x + pw] += win[:ph, :pw]

    for i in range(0, len(jobs), batch_size):
        run(jobs[i : i + batch_size])

    out = np.divide(acc, weight, out=np.zeros_like(acc), where=weight > 0)
    if mask is not None:
        out[~mask] = 0.0
    return out


def best_fbeta(prob: np.ndarray, labels: np.ndarray, mask: np.ndarray, beta: float = 0.5, n_bins: int = 200) -> dict:
    """Best F-beta over thresholds (the Vesuvius ink-detection competition used F0.5), restricted to papyrus pixels."""
    p, y = prob[mask], labels[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pos, _ = np.histogram(p[y], bins=edges)
    neg, _ = np.histogram(p[~y], bins=edges)
    tp = np.cumsum(pos[::-1])[::-1].astype(np.float64)  # predicted ink at threshold edges[i]
    fp = np.cumsum(neg[::-1])[::-1].astype(np.float64)
    fn = pos.sum() - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    b2 = beta**2
    denom = b2 * precision + recall
    f = np.divide((1 + b2) * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    best = int(np.argmax(f))
    return {
        "f_beta": float(f[best]),
        "threshold": float(edges[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "ink_fraction": float(y.mean()),
    }
