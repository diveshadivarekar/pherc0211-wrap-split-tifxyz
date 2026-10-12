"""Synthetic text-line corpus for M5.

The CT phantom's ink is a handful of arcs, not real glyphs — there is no
genuine character identity to recover from it (see run.py for how the full
pipeline is honest about that). To still deliver a genuinely working CRNN +
lacuna-infilling + translation demonstration (FR-5.1-FR-5.5), this generates
its own self-contained synthetic "manuscript text" corpus: short strings
from a small fixed vocabulary, rendered as text-line images with a system
font, that the OCR/infilling models train and are evaluated against
end-to-end (CER/WER, per SRS Test Plan T7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VOCAB = " abcdefghijklmnopqrstuvwxyz"
BLANK_IDX = 0  # CTC blank
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
N_CLASSES = len(VOCAB) + 1  # + blank

WORDS = [
    "scroll", "papyrus", "ink", "page", "text", "ancient", "book", "read",
    "layer", "fused", "carbon", "light", "scan", "volume", "trace", "word",
    "line", "mark", "sign", "letter", "roll", "wrap", "leaf", "grain",
    "fold", "seal", "edge", "space", "gap", "form",
]

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def sample_text(rng: np.random.Generator, n_words: tuple[int, int] = (1, 3)) -> str:
    k = int(rng.integers(n_words[0], n_words[1] + 1))
    return " ".join(rng.choice(WORDS) for _ in range(k))


def render_text_line(text: str, height: int = 32, font_size: int = 24) -> np.ndarray:
    """Render ``text`` as a grayscale ink-on-blank line image, ink normalised to [0, 1]."""
    font = _load_font(font_size)
    dummy = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(bbox[2] - bbox[0] + 8, 16)

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((4, (height - font_size) // 2 - bbox[1]), text, font=font, fill=0)

    arr = 1.0 - (np.asarray(img, dtype=np.float32) / 255.0)  # ink = 1, blank = 0
    return arr


def text_to_indices(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX], dtype=np.int64)


def generate_corpus(
    n_samples: int, rng: np.random.Generator, height: int = 32, font_size: int = 24
) -> list[tuple[np.ndarray, str]]:
    """Returns a list of (image, text) pairs."""
    samples = []
    for _ in range(n_samples):
        text = sample_text(rng)
        image = render_text_line(text, height=height, font_size=font_size)
        samples.append((image, text))
    return samples
