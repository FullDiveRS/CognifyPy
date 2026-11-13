"""Multilingual: script-aware language detection + phrasebook responses."""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple

from ..core.stimulus import Stimulus
from ..util import tokenize
from .base import Capability, LanguagePercept

# Characteristic function words per language (weight-bearing, hand-curated).
PROFILES: Dict[str, List[str]] = {
    "en": ["the", "is", "and", "you", "that", "have", "what", "this", "hello", "with", "how"],
    "es": ["el", "la", "los", "las", "que", "es", "de", "y", "hola", "gracias", "una", "por", "cómo", "estás"],
    "fr": ["le", "la", "les", "est", "et", "de", "que", "bonjour", "merci", "je", "une", "comment", "ça", "va"],
    "de": ["der", "die", "das", "und", "ist", "nicht", "hallo", "danke", "ich", "ein", "eine", "gut", "wie"],
    "it": ["il", "la", "che", "di", "sono", "ciao", "grazie", "un", "una", "e", "buongiorno", "come", "stai"],
    "pt": ["o", "a", "os", "que", "de", "olá", "obrigado", "uma", "não", "como", "você", "está", "bem"],
    "nl": ["de", "het", "een", "en", "ik", "hallo", "dank", "dat", "niet", "je", "hoe", "gaat"],
    "ru": ["и", "в", "не", "что", "привет", "спасибо", "это", "на", "как", "дела", "ты"],
    "zh": ["你", "好", "谢谢", "的", "是", "不", "我", "吗", "很"],
    "ja": ["こんにちは", "ありがとう", "です", "ます", "の", "はい", "いいえ", "なに"],
    "ko": ["안녕", "감사", "합니다", "이", "그", "저", "세요"],
}

PHRASEBOOK: Dict[str, Dict[str, str]] = {
    "greeting": {"en": "Hello", "es": "¡Hola", "fr": "Bonjour", "de": "Hallo",
                 "it": "Ciao", "pt": "Olá", "nl": "Hallo", "ru": "Привет",
                 "zh": "你好", "ja": "こんにちは", "ko": "안녕하세요"},
    "thanks": {"en": "Thank you", "es": "Gracias", "fr": "Merci", "de": "Danke",
               "it": "Grazie", "pt": "Obrigado", "nl": "Dank je", "ru": "Спасибо",
               "zh": "谢谢", "ja": "ありがとう", "ko": "감사합니다"},
    "bye": {"en": "Goodbye", "es": "Adiós", "fr": "Au revoir", "de": "Auf Wiedersehen",
            "it": "Arrivederci", "pt": "Adeus", "nl": "Tot ziens", "ru": "До свидания",
            "zh": "再见", "ja": "さようなら", "ko": "안녕히 가세요"},
}

MINI_DICT: Dict[str, Dict[str, str]] = {
    "hello": {"es": "hola", "fr": "bonjour", "de": "hallo", "it": "ciao", "pt": "olá",
              "nl": "hallo", "ru": "привет", "zh": "你好", "ja": "こんにちは", "ko": "안녕하세요"},
    "thanks": {"es": "gracias", "fr": "merci", "de": "danke", "it": "grazie",
               "pt": "obrigado", "nl": "dank je", "ru": "спасибо", "zh": "谢谢",
               "ja": "ありがとう", "ko": "감사합니다"},
    "friend": {"es": "amigo", "fr": "ami", "de": "freund", "it": "amico",
               "pt": "amigo", "ru": "друг", "zh": "朋友", "ja": "友達", "ko": "친구"},
    "water": {"es": "agua", "fr": "eau", "de": "wasser", "it": "acqua", "pt": "água",
              "ru": "вода", "zh": "水", "ja": "水", "ko": "물"},
    "cat": {"es": "gato", "fr": "chat", "de": "katze", "it": "gatto", "pt": "gato",
            "ru": "кот", "zh": "猫", "ja": "猫", "ko": "고양이"},
}


def detect_script(text: str) -> str:
    """Coarse unicode-script detection."""
    counts = {"han": 0, "kana": 0, "hangul": 0, "cyrillic": 0, "latin": 0}
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            counts["han"] += 1
        elif 0x3040 <= o <= 0x30FF:
            counts["kana"] += 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            counts["hangul"] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A or 0xC0 <= o <= 0x24F:
            counts["latin"] += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "unknown"
    if best == "han" and counts["kana"] > 0:
        return "kana"
    return best


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


class MultilingualCapability(Capability):
    name = "multilingual"
    description = "Detects input language and greets in kind (11 languages)."
    phase = "perceive"

    def __init__(self) -> None:
        self._token_profiles = {lang: [w.lower() for w in words]
                                for lang, words in PROFILES.items()}

    # -- detection ------------------------------------------------------------
    def detect(self, text: str) -> Tuple[str, float]:
        """Returns (language, confidence 0..1). 'unknown' when unsure."""
        if not text or not text.strip():
            return "unknown", 0.0
        script = detect_script(text)
        if script == "kana":
            return "ja", 0.95
        if script == "hangul":
            return "ko", 0.95
        if script == "cyrillic":
            return "ru", 0.9
        if script == "han":
            return "zh", 0.9

        tokens = [t.lower() for t in tokenize(_strip_accents(text))]
        raw_tokens = [t.lower() for t in text.split()]
        scores: Dict[str, float] = {}
        for lang, words in self._token_profiles.items():
            score = 0.0
            for w in words:
                if " " not in w and (w in tokens or w in raw_tokens or
                                     any(t.startswith(w) and len(w) > 3 for t in tokens)):
                    score += 2.0
            scores[lang] = score
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_lang, best = ranked[0]
        second = ranked[1][1]
        if best == 0.0:
            return "en", 0.3          # latin script default, low confidence
        confidence = min(0.95, 0.55 + 0.12 * best + 0.05 * (best - second))
        return best_lang, confidence

    # -- phrase helpers ----------------------------------------------------------
    def greet(self, lang: str) -> str:
        return PHRASEBOOK["greeting"].get(lang, PHRASEBOOK["greeting"]["en"])

    def phrase(self, phrase: str, lang: str) -> str:
        return PHRASEBOOK.get(phrase, {}).get(lang, "")

    def translate(self, word: str, lang: str) -> str:
        return MINI_DICT.get(word.lower(), {}).get(lang, "")

    # -- capability API ------------------------------------------------------------
    def can_handle(self, stimulus: Stimulus, ctx=None) -> bool:
        return bool(stimulus.text)

    def handle(self, stimulus: Stimulus, ctx=None) -> LanguagePercept:
        lang, conf = self.detect(stimulus.text)
        if ctx is not None:
            ctx.scratch["language"] = lang
            if lang != "en" and conf >= 0.5:
                ctx.scratch["localized_greeting"] = self.greet(lang)
        script = detect_script(stimulus.text)
        return LanguagePercept(
            description=(f"Detected language: {lang} ({script} script, "
                         f"confidence {conf:.2f})."),
            confidence=conf, language=lang, script=script,
        )
