from __future__ import annotations

import re
import threading
from functools import lru_cache

LANGUAGE_CODES = {"English": "en", "Turkish": "tr", "Arabic": "ar", "French": "fr", "Italian": "it"}
TURKISH_CHARACTERS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
TURKISH_MARKERS = re.compile(
    r"\b(ad[iı]n|nerede|biliyor|biliyorsun|musun|mısın|misin|müsün|kaç|kac|"
    r"yıllık|yillik|deneyim|tecrübe|tecrube|eğitim|egitim|ikamet|hakkında|hakkinda|şirket|"
    r"sirket|çalıştın|calistin|projeler|ile ilgili|ne yaptın|ne yaptin|rabbin|dinin|peygamberin)\b",
    re.IGNORECASE,
)
FRENCH_MARKERS = re.compile(r"[àâçéèêëîïôùûüÿœ]|\b(quel|quelle|où|avez|vous|expérience|travail)\b", re.IGNORECASE)
ITALIAN_MARKERS = re.compile(r"\b(come|dove|quale|quali|conosci|esperienza|lavorato|formazione)\b", re.IGNORECASE)
ARABIC_MARKERS = re.compile(r"[\u0600-\u06ff]")


def detect_language(text: str) -> str:
    if ARABIC_MARKERS.search(text):
        return "Arabic"
    if TURKISH_CHARACTERS.search(text) or TURKISH_MARKERS.search(text):
        return "Turkish"
    if FRENCH_MARKERS.search(text):
        return "French"
    if ITALIAN_MARKERS.search(text):
        return "Italian"
    return "English"


class LocalTranslationLayer:
    """Offline English-pivot translation backed by separately installed Argos models."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @staticmethod
    @lru_cache(maxsize=32)
    def _get(source_code: str, target_code: str):
        from argostranslate import translate

        translation = translate.get_translation_from_codes(source_code, target_code)
        if translation is None:
            raise RuntimeError(f"Offline translation package is missing: {source_code}->{target_code}")
        return translation

    @lru_cache(maxsize=2048)
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text or source_language == target_language:
            return text
        with self._lock:
            return self._get(LANGUAGE_CODES[source_language], LANGUAGE_CODES[target_language]).translate(text).strip()

    def to_english(self, text: str, source_language: str) -> str:
        return self.translate(text, source_language, "English")

    def from_english(self, text: str, target_language: str) -> str:
        return self.translate(text, "English", target_language)

    def status(self) -> dict[str, object]:
        from argostranslate import translate

        installed = []
        for language, code in LANGUAGE_CODES.items():
            if code != "en" and translate.get_translation_from_codes(code, "en") and translate.get_translation_from_codes("en", code):
                installed.append(language)
        return {"engine": "argos", "offline": True, "english_pivot_languages": installed}


translator = LocalTranslationLayer()
