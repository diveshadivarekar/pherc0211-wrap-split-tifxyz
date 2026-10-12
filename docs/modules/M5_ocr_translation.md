# M5 - OCR, contextual error-correction and translation (PHerc. 0211, scan 20250821151803)

## Role in the pipeline
Turns an ink-confidence map into transcribed, confidence-annotated, optionally translated text. Architecture: CRNN + CTC with per-token log-likelihood,
a masked-language-model infilling step for lacunae, a translation step, and the rule **NFR-6: a low-confidence reading is never presented as text**.

## Status on PHerc. 0211: not run - by design
* M4 produced no ink map that survives the shifted-surface controls (`M4_ink_detection.md`). There is no text-bearing region to transcribe.
* Feeding blob-shaped probability maps to a recogniser would produce plausible-looking characters that the CT does not support - the failure the
  organizers warn about. The correct output of this module for this scroll today is **no transcript**, and that is what the gate is for.
* No lines, letters, readings or translations are given or implied anywhere in this package.

## What exists
* Code: `code/src/m5_ocr_translation/`: CRNN/CTC recogniser trained on rendered synthetic text, a masked-character infilling model, a **dictionary
  stub for translation** (not a trained NMT model), CER/WER, and a pipeline that flags tokens below a confidence threshold.
* Tests: `code/tests/test_m5_ocr_translation.py` pass (synthetic glyphs/text only: training loss decreases, low-confidence flagging, CER/WER, end-to-end
  page transcription on synthetic pages). They say nothing about papyrus, and no Greek/Latin recogniser or real translation model is part of this package.
* Interface: input = an `ink_probability` page; output = confidence-annotated text.

## Would unlock it
A held-out, validated ink signal on a region of 0211 (e.g. shifted-surface-controlled output with stroke-like structure that survives flattening
changes, and ideally a labelled region from the organizers' or community's tooling). Until then M5 has no 0211 result.
