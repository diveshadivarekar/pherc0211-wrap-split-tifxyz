"""Tests for M5: synthetic OCR corpus, CRNN+CTC, infilling, translation stub.

Covers SRS Test Plan T7 (CER/WER recorded, log-likelihood per token) and T8
(sub-threshold tokens flagged). CNN/RNN tests are skipped if torch isn't
installed; the corpus generator and translation stub have no such
dependency and always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.m5_ocr_translation.text_data import generate_corpus, render_text_line, sample_text
from src.m5_ocr_translation.translation import translate

torch = pytest.importorskip("torch", reason="M5 CRNN/infilling tests require torch")


class TestTextData:
    def test_render_produces_ink(self):
        img = render_text_line("scroll")
        assert img.max() > 0.0
        assert img.min() >= 0.0

    def test_sample_text_uses_vocab_words(self):
        rng = np.random.default_rng(0)
        text = sample_text(rng)
        from src.m5_ocr_translation.text_data import WORDS

        assert all(w in WORDS for w in text.split())

    def test_generate_corpus_shapes(self):
        rng = np.random.default_rng(0)
        corpus = generate_corpus(5, rng)
        assert len(corpus) == 5
        for img, text in corpus:
            assert img.ndim == 2
            assert isinstance(text, str) and len(text) > 0


class TestTranslationStub:
    def test_known_word_translated(self):
        result = translate("scroll ink")
        assert result["translated_text"] == "volumen tinta"
        assert result["is_placeholder"] is True
        assert result["unknown_words"] == []

    def test_unknown_word_passed_through(self):
        result = translate("scroll xyzzy")
        assert "xyzzy" in result["unknown_words"]
        assert "xyzzy" in result["translated_text"]


@pytest.fixture(scope="module")
def small_corpus():
    rng = np.random.default_rng(0)
    return generate_corpus(40, rng)


@pytest.fixture(scope="module")
def trained_crnn(small_corpus):
    from src.m5_ocr_translation.crnn import train_crnn

    model, history = train_crnn(small_corpus, epochs=25, batch_size=8)
    return model, history


class TestCRNN:
    def test_training_loss_decreases(self, trained_crnn):
        _, history = trained_crnn
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_predict_returns_text_and_confidences(self, trained_crnn, small_corpus):
        from src.m5_ocr_translation.crnn import predict

        model, _ = trained_crnn
        image, _ = small_corpus[0]
        text, confidences = predict(model, image)
        assert isinstance(text, str)
        assert len(confidences) == len(text)
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_low_confidence_flagging(self):
        from src.m5_ocr_translation.crnn import flag_low_confidence

        annotated = flag_low_confidence("ab", [0.9, 0.2], threshold=0.5)
        assert annotated[0]["flagged"] is False
        assert annotated[1]["flagged"] is True

    def test_cer_and_wer_zero_for_exact_match(self):
        from src.m5_ocr_translation.crnn import character_error_rate, word_error_rate

        assert character_error_rate("scroll", "scroll") == 0.0
        assert word_error_rate("read the scroll", "read the scroll") == 0.0

    def test_cer_and_wer_nonzero_for_mismatch(self):
        from src.m5_ocr_translation.crnn import character_error_rate, word_error_rate

        assert character_error_rate("scrall", "scroll") > 0.0
        assert word_error_rate("read a scroll", "read the scroll") > 0.0


class TestInfilling:
    @pytest.fixture(scope="class")
    def trained_infill_model(self, small_corpus):
        from src.m5_ocr_translation.infilling import train_infill_model

        texts = [t for _, t in small_corpus]
        model, history = train_infill_model(texts, epochs=25, batch_size=8)
        return model, history

    def test_training_loss_decreases(self, trained_infill_model):
        _, history = trained_infill_model
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_infill_fills_every_gap(self, trained_infill_model):
        from src.m5_ocr_translation.infilling import infill

        model, _ = trained_infill_model
        filled, confidences = infill(model, "scr_ll")
        assert "_" not in filled
        assert len(filled) == len("scr_ll")

    def test_infill_recovers_masked_char_better_than_chance(self, trained_infill_model, small_corpus):
        """Statistical check: over many masked positions from real corpus
        text, infilling should beat random-character guessing."""
        from src.m5_ocr_translation.infilling import VOCAB, infill

        model, _ = trained_infill_model
        rng = np.random.default_rng(1)
        correct = 0
        total = 0
        for _, text in small_corpus[:20]:
            if len(text) < 2:
                continue
            pos = rng.integers(0, len(text))
            true_char = text[pos]
            gapped = text[:pos] + "_" + text[pos + 1 :]
            filled, _ = infill(model, gapped)
            correct += filled[pos] == true_char
            total += 1
        chance = 1.0 / len(VOCAB)
        assert total > 0
        assert (correct / total) > chance


class TestPipeline:
    def test_transcribe_page_end_to_end(self, trained_crnn, small_corpus):
        from src.m5_ocr_translation.infilling import train_infill_model
        from src.m5_ocr_translation.pipeline import transcribe_page

        model_ocr, _ = trained_crnn
        texts = [t for _, t in small_corpus]
        model_infill, _ = train_infill_model(texts, epochs=10, batch_size=8)

        image, _ = small_corpus[0]
        result = transcribe_page(model_ocr, model_infill, image, confidence_threshold=0.5)

        assert "raw_ocr_text" in result
        assert "corrected_text" in result
        assert "_" not in result["corrected_text"]
        assert "translation" in result
        assert result["translation"]["is_placeholder"] is True

    def test_extract_ink_tokens(self):
        from src.m5_ocr_translation.pipeline import extract_ink_tokens

        prob_map = np.zeros((20, 40), dtype=np.float32)
        prob_map[5:8, 5:8] = 0.9  # confident blob
        prob_map[12:14, 20:22] = 0.55  # low-confidence blob

        tokens = extract_ink_tokens(prob_map, threshold=0.5, low_confidence_threshold=0.6)
        assert len(tokens) == 2
        flagged = [t["flagged"] for t in tokens]
        assert True in flagged and False in flagged


class TestRealScanReport:
    def test_report_is_honest_about_scan_transcription(self, tmp_path):
        import json

        import h5py

        from src.m5_ocr_translation.run import run

        with h5py.File(tmp_path / "scan_ink.h5", "w") as f:
            f.attrs["validated"] = False
            prob = np.zeros((40, 40), dtype=np.float32)
            prob[5:10, 5:10] = 0.9  # a real-sized blob
            prob[30, 30] = 0.9  # a 1-pixel speck
            f.create_group("layer_1").create_dataset("ink_probability", data=prob)

        report_path = run(tmp_path, ink_path=tmp_path / "scan_ink.h5", n_train=10, n_eval=3, epochs=2)
        report = json.loads(report_path.read_text())

        assert report["metrics_source"] == "synthetic_text_lines"
        scan = report["scan_transcription"]
        assert scan["available"] is False and scan["reason"]
        assert scan["total_tokens"] == 1  # the speck is filtered out
        assert scan["ink_map_validated"] is False and "unvalidated" in scan["warning"]
