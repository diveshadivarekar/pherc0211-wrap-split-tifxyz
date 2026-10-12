"""M5 — Ithaca-style contextual lacuna infilling.

Implements FR-5.3: fill masked/low-confidence character positions using
surrounding context, rather than leaving a gap or asserting an unsupported
guess. A small bidirectional char-level LSTM stands in for the transformer
masked-language model in Assael et al.'s Ithaca (ARCHITECTURE.md's
reference) — same task (predict a masked token from both-direction
context), much smaller model, trained on the same synthetic corpus as the
CRNN (text_data.py).
"""

from __future__ import annotations

import numpy as np

from src.m5_ocr_translation.text_data import CHAR_TO_IDX, IDX_TO_CHAR, VOCAB

MASK_IDX = len(VOCAB)  # one extra "mask" symbol appended to the vocab
N_INFILL_CLASSES = len(VOCAB) + 1


def build_infill_model(hidden_size: int = 64):
    import torch.nn as nn

    class MaskedInfillLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(N_INFILL_CLASSES, 32)
            self.rnn = nn.LSTM(32, hidden_size, bidirectional=True, batch_first=True)
            self.head = nn.Linear(hidden_size * 2, len(VOCAB))

        def forward(self, x):
            # x: (batch, seq_len) int indices, MASK_IDX marks the gap(s)
            emb = self.embed(x)
            out, _ = self.rnn(emb)
            return self.head(out)  # (batch, seq_len, len(VOCAB)) — logits over real chars only

    return MaskedInfillLSTM()


def _text_to_infill_indices(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX[c] - 1 for c in text if c in CHAR_TO_IDX], dtype=np.int64)


def _make_training_example(text: str, rng: np.random.Generator, mask_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (input_indices_with_mask, target_indices, mask_positions)."""
    target = _text_to_infill_indices(text)
    n_mask = max(1, int(len(target) * mask_frac))
    mask_positions = rng.choice(len(target), size=min(n_mask, len(target)), replace=False)

    inputs = target.copy()
    inputs[mask_positions] = MASK_IDX
    return inputs, target, mask_positions


def train_infill_model(
    texts: list[str],
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    mask_frac: float = 0.2,
    device: str | None = None,
    seed: int = 0,
):
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_infill_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    rng = np.random.default_rng(seed)

    history: dict[str, list[float]] = {"train_loss": []}

    for epoch in range(epochs):
        order = rng.permutation(len(texts))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(texts), batch_size):
            batch_texts = [texts[i] for i in order[start : start + batch_size]]
            examples = [_make_training_example(t, rng, mask_frac) for t in batch_texts]
            examples = [e for e in examples if len(e[0]) > 0]
            if not examples:
                continue

            max_len = max(len(e[0]) for e in examples)
            batch_in = np.zeros((len(examples), max_len), dtype=np.int64)
            batch_target = np.full((len(examples), max_len), -100, dtype=np.int64)
            for i, (inp, tgt, mask_pos) in enumerate(examples):
                batch_in[i, : len(inp)] = inp
                batch_target[i, mask_pos] = tgt[mask_pos]

            x = torch.from_numpy(batch_in).to(device)
            y = torch.from_numpy(batch_target).to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

    model.eval()
    return model, history


def infill(model, text_with_gaps: str, gap_char: str = "_", device: str | None = None) -> tuple[str, list[float]]:
    """Fill every ``gap_char`` in ``text_with_gaps`` using bidirectional
    context. Returns ``(filled_text, per_position_confidence)`` (confidence
    is 1.0 for characters that weren't gaps)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    indices = []
    gap_positions = []
    for i, c in enumerate(text_with_gaps):
        if c == gap_char:
            indices.append(MASK_IDX)
            gap_positions.append(i)
        elif c in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[c] - 1)
        else:
            indices.append(MASK_IDX)
            gap_positions.append(i)

    x = torch.from_numpy(np.array(indices, dtype=np.int64)[np.newaxis]).to(device)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=-1)

    chars = list(text_with_gaps)
    confidences = [1.0] * len(chars)
    for pos in gap_positions:
        best_idx = int(probs[pos].argmax())
        chars[pos] = IDX_TO_CHAR[best_idx + 1]
        confidences[pos] = float(probs[pos, best_idx])

    return "".join(chars), confidences
