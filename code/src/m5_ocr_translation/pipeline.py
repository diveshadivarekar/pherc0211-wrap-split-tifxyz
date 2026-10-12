"""M5 pipeline glue: OCR -> confidence flagging -> infilling -> translation.

Two entry points:

- :func:`transcribe_page` — the genuine, testable OCR+correction+translation
  demo, run on synthetic text-line images (text_data.py) that have real
  ground-truth characters.
- :func:`extract_ink_tokens` — what the M4 pipeline output actually is: ink
  probability blobs on a flattened CT page with no character identity. This
  reports confidence-scored *token regions* (FR-5.2/NFR-6's flagging, honestly
  applied) without inventing what letter each blob "is".
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from src.m5_ocr_translation.crnn import flag_low_confidence, predict
from src.m5_ocr_translation.infilling import infill
from src.m5_ocr_translation.translation import translate


def transcribe_page(
    model_ocr,
    model_infill,
    image: np.ndarray,
    confidence_threshold: float = 0.5,
    target_language: str = "toy",
) -> dict:
    """Full M5 chain on one text-line image with known ground truth
    characters: OCR -> flag low-confidence chars -> infill them from context
    -> translate. Every stage's output is kept, not just the final text, so
    a caller (M6) can show the reader exactly what was recovered vs. guessed.
    """
    raw_text, confidences = predict(model_ocr, image)
    annotated = flag_low_confidence(raw_text, confidences, threshold=confidence_threshold)

    gapped = "".join("_" if a["flagged"] else a["char"] for a in annotated)
    if "_" in gapped:
        filled_text, infill_confidences = infill(model_infill, gapped)
    else:
        filled_text, infill_confidences = raw_text, confidences

    translation = translate(filled_text, target_language=target_language)

    return {
        "raw_ocr_text": raw_text,
        "per_char": annotated,
        "n_flagged": sum(a["flagged"] for a in annotated),
        "corrected_text": filled_text,
        "infill_confidences": infill_confidences,
        "translation": translation,
    }


def extract_ink_tokens(
    ink_prob_map: np.ndarray, threshold: float = 0.5, min_pixels: int = 2, low_confidence_threshold: float = 0.6
) -> list[dict]:
    """Group a flattened page's ink-probability map into candidate ink
    tokens (connected blobs above ``threshold``) with a per-token confidence
    and NFR-6 low-confidence flag.

    Deliberately does **not** assign a character identity — the CT phantom's
    ink is procedurally-placed arcs, not real glyphs, and asserting "this
    blob is the letter X" would be exactly the kind of unsupported guess
    NFR-6 forbids. A real deployment would feed each token's bounding-box
    crop from the flattened page into :func:`crnn.predict` instead.
    """
    binary = ink_prob_map > threshold
    labels, n = ndi.label(binary)

    tokens = []
    for token_id in range(1, n + 1):
        ys, xs = np.where(labels == token_id)
        if len(ys) < min_pixels:
            continue
        mean_conf = float(ink_prob_map[ys, xs].mean())
        tokens.append(
            {
                "bbox": [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1],
                "n_pixels": int(len(ys)),
                "mean_confidence": mean_conf,
                "flagged": mean_conf < low_confidence_threshold,
            }
        )
    return tokens
