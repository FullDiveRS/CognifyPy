"""Emotions: Plutchik's eight primaries with decay dynamics, plus the
finer-grained feelings from Darwin's taxonomy (love, devotion, tenderness,
suffering, weeping, high/low spirits, anxiety, grief, dejection, despair,
hatred, disdain, contempt, guilt, pride, helplessness, patience,
affirmation, negation, self-attention, shyness, modesty, blushing,
reflection, meditation, ill-temper, sulkiness, determination)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

from ..util import clamp

EMOTION_NAMES: Tuple[str, ...] = (
    "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation",
    # Darwin's finer-grained feelings — APPENDED so persisted snapshots
    # (stored by name) keep loading correctly.
    "love", "devotion", "tenderness", "suffering", "weeping",
    "high spirits", "low spirits", "anxiety", "grief", "dejection",
    "despair", "hatred", "disdain", "contempt", "guilt", "pride",
    "helplessness", "patience", "affirmation", "negation",
    "self attention", "shyness", "modesty", "blushing", "reflection",
    "meditation", "ill temper", "sulkiness", "determination",
)

POSITIVE_EMOTIONS = frozenset({
    "joy", "trust",
    "love", "devotion", "tenderness", "high spirits", "pride",
    "determination", "patience", "affirmation", "modesty",
})
NEGATIVE_EMOTIONS = frozenset({
    "fear", "sadness", "disgust", "anger",
    "suffering", "weeping", "low spirits", "anxiety", "grief", "dejection",
    "despair", "hatred", "disdain", "contempt", "guilt", "helplessness",
    "ill temper", "sulkiness", "negation",
})

# Interjection used by the response composer for each dominant emotion.
INTERJECTIONS: Dict[str, str] = {
    "joy": "That makes me happy —",
    "trust": "I hear you —",
    "fear": "I feel a little cautious —",
    "surprise": "Oh, interesting —",
    "sadness": "That weighs on me —",
    "disgust": "Hmm, that doesn't sit well —",
    "anger": "That frustrates me —",
    "anticipation": "I'm curious about this —",
    # Darwin's finer feelings
    "love": "That warms my heart —",
    "devotion": "I feel steadfast —",
    "tenderness": "I feel a gentle warmth —",
    "suffering": "This is hard to bear —",
    "weeping": "I feel tearful —",
    "high spirits": "I feel buoyant —",
    "low spirits": "I feel a bit low —",
    "anxiety": "I feel uneasy —",
    "grief": "This weighs deeply on me —",
    "dejection": "I feel deflated —",
    "despair": "I feel hopeless about this —",
    "hatred": "I feel a burning dislike —",
    "disdain": "I find this beneath respect —",
    "contempt": "I feel dismissive —",
    "guilt": "I feel responsible —",
    "pride": "I feel quietly proud —",
    "helplessness": "I feel stuck —",
    "patience": "I can wait calmly —",
    "affirmation": "I agree —",
    "negation": "I disagree —",
    "self attention": "I feel seen —",
    "shyness": "I feel a little shy —",
    "modesty": "I feel humbled —",
    "blushing": "I am blushing —",
    "reflection": "Let me reflect —",
    "meditation": "I am absorbed in thought —",
    "ill temper": "I feel peevish —",
    "sulkiness": "I feel sulky —",
    "determination": "I am determined —",
}


class EmotionalState:
    """All EMOTION_NAMES (Plutchik's primaries + Darwin's finer feelings),
    each in [0, 1], with per-cycle decay."""

    def __init__(self, sensitivity: float = 1.0, initial: Optional[Dict[str, float]] = None):
        self.sensitivity = max(0.1, min(3.0, float(sensitivity)))
        self._values: Dict[str, float] = {name: 0.0 for name in EMOTION_NAMES}
        if initial:
            for k, v in initial.items():
                if k in self._values:
                    self._values[k] = clamp(float(v))

    # -- core dynamics -------------------------------------------------------
    def apply(self, deltas: Dict[str, float], scale: float = 1.0) -> None:
        """Add emotion deltas (already signed magnitudes in [0, 1])."""
        s = self.sensitivity * scale
        for name, delta in deltas.items():
            if name in self._values:
                self._values[name] = clamp(self._values[name] + delta * s)

    def decay(self, rate: float = 0.08) -> None:
        """Passive emotional cooling each cycle."""
        for name in self._values:
            self._values[name] = max(0.0, self._values[name] * (1.0 - rate))

    def get(self, name: str) -> float:
        return self._values.get(name, 0.0)

    def boost(self, name: str, amount: float) -> None:
        if name in self._values:
            self._values[name] = clamp(self._values[name] + amount)

    # -- aggregates -----------------------------------------------------------
    @property
    def values(self) -> Dict[str, float]:
        return dict(self._values)

    @property
    def valence(self) -> float:
        """-1 (negative) .. +1 (positive)."""
        pos = sum(self._values[e] for e in POSITIVE_EMOTIONS)
        neg = sum(self._values[e] for e in NEGATIVE_EMOTIONS)
        return clamp(pos - neg, -1.0, 1.0)

    @property
    def arousal(self) -> float:
        """Overall emotional energy 0..1."""
        return clamp(sum(self._values.values()) / len(EMOTION_NAMES) * 2.0)

    def dominant(self, threshold: float = 0.15) -> Optional[str]:
        """Name of the strongest active emotion, or None if all quiet."""
        best, best_v = None, threshold
        for name, v in self._values.items():
            if v > best_v:
                best, best_v = name, v
        return best

    def snapshot(self) -> Dict[str, float]:
        return dict(self._values)

    def describe(self) -> str:
        """Human-readable description, e.g. "joy (0.42) | mood positive (+0.30)".
        Mirrors the C# EmotionalState.Describe()."""
        active = sorted((kv for kv in self._values.items() if kv[1] > 0.08),
                        key=lambda kv: -kv[1])[:3]
        base = ", ".join(f"{k} ({v:.2f})" for k, v in active) or "calm"
        v = self.valence
        return f"{base} | mood {'positive' if v >= 0 else 'negative'} ({v:+.2f})"

    def load(self, snapshot: Dict[str, float]) -> None:
        for k, v in snapshot.items():
            if k in self._values:
                self._values[k] = clamp(float(v))

    def __str__(self) -> str:
        active = {k: round(v, 2) for k, v in self._values.items() if v > 0.01}
        return f"EmotionalState({active or 'calm'})"
