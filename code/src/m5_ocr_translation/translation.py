"""M5 — translation into modern languages (FR-5.5).

This is a **placeholder interface, not a real translator**. A genuine
neural machine translation model (per ARCHITECTURE.md) needs a bilingual
corpus and pretrained weights (e.g. a MarianMT/NLLB checkpoint) that this
offline PoC environment doesn't have, and the synthetic corpus in
text_data.py isn't real language anyway — there's nothing authentic to
translate. ``translate()`` below applies a deterministic, declared
substitution table so the *pipeline wiring* (confidence-annotated text in,
translated text + confidence out, per TBD-1 target-language config) is real
and testable, while the linguistic content is explicitly a toy.

Swap ``translate()``'s body for a real model call (e.g. a HuggingFace
``transformers`` MarianMTModel) when one is available; the signature is
already the integration point M6 calls.
"""

from __future__ import annotations

# A tiny, explicitly-fake "toy language" substitution table so the
# translation stage of the pipeline is exercised end-to-end without
# pretending to know a real target language.
_TOY_DICTIONARY = {
    "scroll": "volumen", "papyrus": "papiro", "ink": "tinta", "page": "pagina",
    "text": "texto", "ancient": "antiguo", "book": "libro", "read": "leer",
    "layer": "capa", "fused": "fusionado", "carbon": "carbono", "light": "luz",
    "scan": "escaneo", "volume": "volumen", "trace": "rastro", "word": "palabra",
    "line": "linea", "mark": "marca", "sign": "signo", "letter": "letra",
    "roll": "rollo", "wrap": "envoltura", "leaf": "hoja", "grain": "grano",
    "fold": "pliegue", "seal": "sello", "edge": "borde", "space": "espacio",
    "gap": "brecha", "form": "forma",
}


def translate(text: str, target_language: str = "toy") -> dict:
    """Translate ``text`` word-by-word via the toy dictionary.

    Returns ``{"translated_text": str, "target_language": str, "is_placeholder": True,
    "unknown_words": [...]}`` — ``unknown_words`` lists words the toy
    dictionary had no entry for (left untranslated), which is exactly the
    kind of gap a real NMT integration needs to handle explicitly rather
    than silently.
    """
    words = text.split()
    translated_words = []
    unknown = []
    for w in words:
        key = w.lower().strip(".,!?")
        if key in _TOY_DICTIONARY:
            translated_words.append(_TOY_DICTIONARY[key])
        else:
            translated_words.append(w)
            if w:
                unknown.append(w)

    return {
        "translated_text": " ".join(translated_words),
        "target_language": target_language,
        "is_placeholder": True,
        "unknown_words": unknown,
    }
