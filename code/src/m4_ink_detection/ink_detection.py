"""M4 — AI ink detection: per-voxel ink-confidence from local 3D texture.

Implements FR-4.1-FR-4.4. ARCHITECTURE.md specifies extracting a localised
3D sub-volume window per pixel (e.g. 64x64x26) and running a 3D CNN /
TimeSformer ensemble over it — that's the interface implemented here, at a
window size that fits this PoC's much smaller phantom (default 7x7x7
instead of 64x64x26) and with a small from-scratch 3D CNN instead of a
pretrained TimeSformer ensemble.

A classical density-threshold baseline (``classical_ink_probability``) is
always available with zero dependencies — this is what real high-Z metallic
ink looks like (per ARCHITECTURE.md's photoelectric-absorption discussion,
metallic ink absorbs enough more than substrate that simple thresholding
already works). The learned model (``train_ink_model`` /
``predict_ink_map``) is what's needed for low-Z carbon ink, where density
alone doesn't separate ink from substrate and the network has to pick up on
subtler local texture.
"""

from __future__ import annotations

import numpy as np

DEFAULT_WINDOW = (7, 7, 7)


def extract_window(volume: np.ndarray, center: tuple[int, int, int], window: tuple[int, int, int]) -> np.ndarray:
    """Extract a local sub-volume centred on ``center`` (z, y, x), zero-padded at borders.

    Pads the *whole* volume on every call — fine for a handful of windows,
    but O(volume size) per call. For many windows (training-data sampling,
    per-pixel inference over a flattened page) use :func:`extract_windows_batch`
    instead, which pads once.
    """
    wz, wy, wx = window
    hz, hy, hx = wz // 2, wy // 2, wx // 2
    z, y, x = center

    padded = np.pad(volume, ((hz, hz), (hy, hy), (hx, hx)), mode="constant", constant_values=0.0)
    return padded[z : z + wz, y : y + wy, x : x + wx]


def extract_windows_batch(
    volume: np.ndarray, centers: np.ndarray, window: tuple[int, int, int]
) -> np.ndarray:
    """Extract many windows efficiently: pad ``volume`` once, then slice.

    ``centers`` is (N, 3) int array of (z, y, x). Equivalent to
    ``np.stack([extract_window(volume, c, window) for c in centers])`` but
    without re-padding the whole volume N times (extract_window's per-call
    padding cost made per-pixel inference over a flattened page, and
    training-window sampling, needlessly re-allocate the full volume
    thousands of times).
    """
    wz, wy, wx = window
    hz, hy, hx = wz // 2, wy // 2, wx // 2
    padded = np.pad(volume, ((hz, hz), (hy, hy), (hx, hx)), mode="constant", constant_values=0.0)

    windows = np.empty((len(centers), wz, wy, wx), dtype=padded.dtype)
    for i, (z, y, x) in enumerate(centers):
        windows[i] = padded[z : z + wz, y : y + wy, x : x + wx]
    return windows


def classical_ink_probability(volume: np.ndarray, substrate_density: float, ink_density: float) -> np.ndarray:
    """Zero-dependency baseline: linearly rescale density into [0, 1] between
    the substrate and ink reference levels — a stand-in for readily-thresholded
    high-Z metallic ink."""
    span = max(ink_density - substrate_density, 1e-6)
    prob = (volume - substrate_density) / span
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def estimate_reference_densities(
    samples: np.ndarray, substrate_quantile: float = 0.5, ink_quantile: float = 0.99
) -> tuple[float, float]:
    """Calibrate the classical baseline from the scan itself.

    The phantom ships its own substrate/ink density levels; a real scan does
    not, and its density scale (normalised, possibly negative) bears no
    relation to the phantom's defaults. ``samples`` is a 1D array of density
    values drawn from valid page pixels: the median stands in for typical
    substrate, a high quantile for the brightest (candidate-ink) material.
    """
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 0.0, 1.0
    substrate = float(np.quantile(samples, substrate_quantile))
    ink = float(np.quantile(samples, ink_quantile))
    if ink - substrate < 1e-6:
        ink = substrate + 1.0
    return substrate, ink


def sample_training_windows(
    volume: np.ndarray,
    ink_mask: np.ndarray,
    foreground_mask: np.ndarray,
    n_positive: int,
    n_negative: int,
    window: tuple[int, int, int] = DEFAULT_WINDOW,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample balanced (window, label) pairs for training.

    Negatives are drawn from ``foreground_mask & ~ink_mask`` (real substrate,
    not empty background) so the model learns ink-vs-substrate texture, not
    just "is there any material here" (which the classical baseline already
    solves).
    """
    rng = rng or np.random.default_rng(0)

    pos_coords = np.argwhere(ink_mask)
    neg_pool = foreground_mask & ~ink_mask
    neg_coords = np.argwhere(neg_pool)

    if len(pos_coords) == 0 or len(neg_coords) == 0:
        return np.empty((0, *window), dtype=np.float32), np.empty((0,), dtype=np.float32)

    pos_idx = rng.integers(0, len(pos_coords), size=min(n_positive, len(pos_coords) * 5))
    neg_idx = rng.integers(0, len(neg_coords), size=min(n_negative, len(neg_coords) * 5))

    centers = np.concatenate([pos_coords[pos_idx], neg_coords[neg_idx]], axis=0)
    labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]).astype(np.float32)
    windows = extract_windows_batch(volume, centers, window).astype(np.float32)

    return windows, labels


def build_ink_model(window: tuple[int, int, int] = DEFAULT_WINDOW):
    """Construct the 3D CNN. Lazy torch import so the classical baseline and
    training-data utilities above work with no ML dependency installed."""
    import torch.nn as nn

    class InkCNN3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, 3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))

        def forward(self, x):
            return self.classifier(self.features(x)).squeeze(-1)

    return InkCNN3D()


def train_ink_model(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
):
    """Train the 3D CNN on sampled windows. Returns ``(model, history)``."""
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ink_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(train_windows[:, np.newaxis]).to(device)
    y = torch.from_numpy(train_labels).to(device)
    n = x.shape[0]

    history: dict[str, list[float]] = {"train_loss": [], "val_accuracy": []}

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(x[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)

        if val_windows is not None and len(val_windows) > 0:
            model.eval()
            with torch.no_grad():
                vx = torch.from_numpy(val_windows[:, np.newaxis]).to(device)
                vy = torch.from_numpy(val_labels).to(device)
                preds = (torch.sigmoid(model(vx)) > 0.5).float()
                acc = (preds == vy).float().mean().item()
            history["val_accuracy"].append(acc)
            model.train()

    model.eval()
    return model, history


def predict_ink_map(
    model,
    volume: np.ndarray,
    coord_map: np.ndarray,
    window: tuple[int, int, int] = DEFAULT_WINDOW,
    device: str | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Run the trained CNN over every valid pixel of a flattened page's
    ``coord_map``, returning a per-pixel ink probability map of the same
    (depth, n_theta) shape."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    h, w = coord_map.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    valid = ~np.isnan(coord_map[:, :, 0])
    coords = np.round(coord_map[valid]).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, volume.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, volume.shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, volume.shape[2] - 1)

    windows = extract_windows_batch(volume, coords, window).astype(np.float32)

    probs = np.empty(len(windows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size, np.newaxis]).to(device)
            probs[start : start + batch_size] = torch.sigmoid(model(batch)).cpu().numpy()

    prob_map[valid] = probs
    return prob_map
