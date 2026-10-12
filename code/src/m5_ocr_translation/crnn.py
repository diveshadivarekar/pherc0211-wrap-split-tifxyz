"""M5 — CRNN + CTC OCR with per-token confidence scoring.

Implements FR-5.1/FR-5.2/NFR-6: transcribe a text-line image, and never
assert a character the model isn't confident about — flag it instead. Torch
is imported lazily so the rest of the package (text_data, translation) has
no hard ML dependency.

Trained and evaluated on the synthetic corpus from text_data.py (see that
module's docstring for why: the CT phantom's ink arcs have no real
character identity to recover, so this validates the OCR+confidence engine
on data that actually has ground-truth text).
"""

from __future__ import annotations

import numpy as np

from src.m5_ocr_translation.text_data import BLANK_IDX, IDX_TO_CHAR, N_CLASSES


def build_crnn():
    import torch.nn as nn

    class CRNN(nn.Module):
        """Conv stack pools height by 8x and width by 4x.

        Width pooling matters as much as it looks unimportant: with *no*
        width pooling, the number of timesteps (image width in pixels, tens
        to hundreds) vastly exceeds the label length (a handful of
        characters), and CTC has a well-known failure mode at large
        timestep:label ratios — predicting blank at every timestep is a
        strong, easy-to-find local optimum ("CTC collapse") that an
        untrained model falls into and then can't escape by gradient
        descent alone. Pooling width down keeps the ratio low enough that
        training actually finds real character predictions instead.

        BatchNorm2d after every conv turned out to matter even more: without
        it, this model reliably collapsed to all-blank even with a healthy
        T/L ratio and hundreds of epochs (verified empirically) — normalising
        each layer's activations was what let gradient descent escape that
        local optimum at all, not just reach it faster.
        """

        def __init__(self, in_height: int = 32, hidden_size: int = 64):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),  # height/2, width unchanged
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),  # height/4, width/2
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),  # height/8, width/4
            )
            feat_height = in_height // 8
            self.width_pool_factor = 4
            self.rnn = nn.LSTM(64 * feat_height, hidden_size, bidirectional=True, batch_first=True)
            self.head = nn.Linear(hidden_size * 2, N_CLASSES)

        def forward(self, x):
            # x: (batch, 1, height, width)
            feats = self.cnn(x)  # (batch, channels, height', width)
            b, c, h, w = feats.shape
            feats = feats.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (batch, time=width, features)
            rnn_out, _ = self.rnn(feats)
            logits = self.head(rnn_out)  # (batch, time, N_CLASSES)
            return logits

    return CRNN()


def _pad_batch(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad a batch of (H, W) images to the same width. Returns (batch, widths)."""
    height = images[0].shape[0]
    max_w = max(img.shape[1] for img in images)
    batch = np.zeros((len(images), 1, height, max_w), dtype=np.float32)
    widths = np.zeros(len(images), dtype=np.int64)
    for i, img in enumerate(images):
        w = img.shape[1]
        batch[i, 0, :, :w] = img
        widths[i] = w
    return batch, widths


def train_crnn(
    samples: list[tuple[np.ndarray, str]],
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: str | None = None,
):
    """Train the CRNN with CTC loss. Returns ``(model, history)``."""
    import torch
    import torch.nn as nn

    from src.m5_ocr_translation.text_data import text_to_indices

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_crnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    history: dict[str, list[float]] = {"train_loss": []}

    for epoch in range(epochs):
        rng_order = np.random.permutation(len(samples))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(samples), batch_size):
            batch_idx = rng_order[start : start + batch_size]
            batch_samples = [samples[i] for i in batch_idx]
            images = [s[0] for s in batch_samples]
            targets = [text_to_indices(s[1]) for s in batch_samples]

            batch_arr, widths = _pad_batch(images)
            x = torch.from_numpy(batch_arr).to(device)

            logits = model(x)  # (batch, time, classes)
            time_steps = logits.shape[1]
            log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (time, batch, classes)

            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            flat_targets = torch.from_numpy(np.concatenate(targets)) if targets else torch.empty(0, dtype=torch.long)
            input_lengths = torch.full((len(batch_samples),), time_steps, dtype=torch.long)

            opt.zero_grad()
            loss = ctc_loss(log_probs.cpu(), flat_targets, input_lengths, target_lengths)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

    model.eval()
    return model, history


def predict(model, image: np.ndarray, device: str | None = None) -> tuple[str, list[float]]:
    """Greedy CTC decode. Returns ``(text, per_char_confidence)`` where
    confidence is exp(log-probability) of the emitted class at the timestep
    that survived blank/repeat collapsing (NFR-6's confidence signal)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    x = torch.from_numpy(image[np.newaxis, np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(x)[0]  # (time, classes)
        probs = torch.softmax(logits, dim=-1)
        best_probs, best_idx = probs.max(dim=-1)

    best_idx = best_idx.cpu().numpy()
    best_probs = best_probs.cpu().numpy()

    chars, confidences = [], []
    prev = None
    for idx, p in zip(best_idx, best_probs):
        if idx != prev and idx != BLANK_IDX:
            chars.append(IDX_TO_CHAR[int(idx)])
            confidences.append(float(p))
        prev = idx

    return "".join(chars), confidences


def flag_low_confidence(text: str, confidences: list[float], threshold: float = 0.5) -> list[dict]:
    """NFR-6: annotate each character with its confidence and whether it
    falls below the configurable threshold — the GUI must never render a
    flagged character as definitive."""
    return [
        {"char": c, "confidence": conf, "flagged": conf < threshold}
        for c, conf in zip(text, confidences)
    ]


def _edit_distance(a: list, b: list) -> int:
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[len(a), len(b)])


def character_error_rate(pred: str, target: str) -> float:
    """Levenshtein edit distance normalised by target length (CER, per T7)."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(list(pred), list(target)) / len(target)


def word_error_rate(pred: str, target: str) -> float:
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred.split()) == 0 else 1.0
    return _edit_distance(pred.split(), target_words) / len(target_words)
