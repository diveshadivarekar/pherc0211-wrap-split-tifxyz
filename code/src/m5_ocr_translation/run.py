"""M5 pipeline entrypoint.

Two things happen here (see pipeline.py for why they're separate):

1. Train + evaluate the CRNN/infilling/translation chain on its own
   synthetic text corpus, reporting CER/WER (SRS Test Plan T7) — this is
   the genuine, ground-truth-backed OCR demonstration.
2. Run the same trained CRNN's confidence machinery over the M4 ink map as
   a live pipeline connection, reporting confidence-scored ink *tokens*
   (not fabricated character identities — see extract_ink_tokens).

Usage:
    python -m src.m5_ocr_translation.run --out data/processed/ \
        [--ink data/processed/phantom_001_ink.h5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# A real scan's ink map is speckled; 2-pixel blobs are noise, not glyph strokes.
DEFAULT_MIN_TOKEN_PIXELS = 12


def run(
    out_dir: str | Path,
    ink_path: str | Path | None = None,
    n_train: int = 150,
    n_eval: int = 30,
    epochs: int = 150,
    confidence_threshold: float = 0.5,
    min_token_pixels: int = DEFAULT_MIN_TOKEN_PIXELS,
) -> Path:
    from src.m5_ocr_translation.crnn import character_error_rate, predict, train_crnn, word_error_rate
    from src.m5_ocr_translation.infilling import train_infill_model
    from src.m5_ocr_translation.pipeline import extract_ink_tokens, transcribe_page
    from src.m5_ocr_translation.text_data import generate_corpus

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    train_corpus = generate_corpus(n_train, rng)
    eval_corpus = generate_corpus(n_eval, np.random.default_rng(1))

    print(f"Training CRNN on {n_train} synthetic text-line images...")
    model_ocr, ocr_history = train_crnn(train_corpus, epochs=epochs)
    print(f"  final train loss: {ocr_history['train_loss'][-1]:.4f}")

    print("Training lacuna-infilling model on the same corpus...")
    model_infill, infill_history = train_infill_model([t for _, t in train_corpus], epochs=epochs)
    print(f"  final train loss: {infill_history['train_loss'][-1]:.4f}")

    cers, wers = [], []
    sample_records = []
    for image, target_text in eval_corpus:
        result = transcribe_page(model_ocr, model_infill, image, confidence_threshold=confidence_threshold)
        cer = character_error_rate(result["corrected_text"], target_text)
        wer = word_error_rate(result["corrected_text"], target_text)
        cers.append(cer)
        wers.append(wer)
        sample_records.append(
            {
                "target": target_text,
                "raw_ocr": result["raw_ocr_text"],
                "corrected": result["corrected_text"],
                "n_flagged": result["n_flagged"],
                "cer": cer,
                "wer": wer,
                "translation": result["translation"]["translated_text"],
            }
        )

    summary = {
        # Top-level CER/WER/samples are kept for existing readers (GUI), but they
        # come from synthetic text-line images only - they say nothing about how
        # well a real scan would be read.
        "n_eval_samples": n_eval,
        "mean_cer": float(np.mean(cers)),
        "mean_wer": float(np.mean(wers)),
        "samples": sample_records[:10],
        "metrics_source": "synthetic_text_lines",
        "scan_transcription": {
            "available": False,
            "reason": (
                "The OCR model is trained only on synthetic text lines; no model for this script/hand "
                "exists, so no characters are asserted for the scan. Only ink-region tokens are reported."
            ),
        },
    }

    if ink_path is not None and Path(ink_path).exists():
        import h5py

        with h5py.File(ink_path, "r") as f:
            ink_validated = bool(f.attrs.get("validated", True))
            ink_summary = {}
            total_tokens = total_flagged = pages_with_ink = 0
            for layer_name in f.keys():
                prob_map = f[layer_name]["ink_probability"][...]
                tokens = extract_ink_tokens(prob_map, min_pixels=min_token_pixels)
                tokens.sort(key=lambda t: -t["n_pixels"])
                n_flagged = sum(t["flagged"] for t in tokens)
                total_tokens += len(tokens)
                total_flagged += n_flagged
                pages_with_ink += bool(tokens)
                ink_summary[layer_name] = {
                    "n_tokens": len(tokens),
                    "n_flagged": n_flagged,
                    "ink_coverage": float((prob_map > 0.5).mean()),
                    "tokens": tokens[:20],
                }
        summary["ink_map_tokens"] = ink_summary
        summary["scan_transcription"].update(
            {
                "n_pages": len(ink_summary),
                "pages_with_ink_tokens": pages_with_ink,
                "total_tokens": total_tokens,
                "total_flagged_low_confidence": total_flagged,
                "ink_map_validated": ink_validated,
            }
        )
        if not ink_validated:
            summary["scan_transcription"]["warning"] = (
                "The ink map is an unvalidated density heuristic (no ink ground truth for this scan); "
                "tokens may be noise, bright inclusions or damage rather than ink."
            )
        print(f"Extracted ink tokens from {len(ink_summary)} flattened pages ({total_tokens} tokens, min {min_token_pixels} px)")

    print(f"Scan transcription: {'available' if summary['scan_transcription']['available'] else 'NOT available'} - {summary['scan_transcription']['reason']}")

    out_path = out_dir / "m5_transcript_report.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"Synthetic-benchmark CER={summary['mean_cer']:.4f}  WER={summary['mean_wer']:.4f} (not a measure of scan reading quality)")
    print(f"Report -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--ink", default=None, help="M4 ink-probability HDF5 (optional)")
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-token-pixels", type=int, default=DEFAULT_MIN_TOKEN_PIXELS, help="Ignore ink blobs smaller than this")
    args = parser.parse_args()

    run(
        args.out,
        ink_path=args.ink,
        n_train=args.n_train,
        n_eval=args.n_eval,
        epochs=args.epochs,
        confidence_threshold=args.confidence_threshold,
        min_token_pixels=args.min_token_pixels,
    )


if __name__ == "__main__":
    main()
