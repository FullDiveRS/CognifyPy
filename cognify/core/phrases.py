"""PhrasePool: a mind's trained vocabulary.

Sayings are sentence-level lines picked up through training that the mind may
speak in its own voice later. This is the direct answer to "she keeps
repeating the same sentence — is that because she does not have many words?":
partly, yes. Every trained saying widens what the mind can say, and the
composer draws from this pool when it has nothing stronger to answer with —
exactly the situation where a small vocabulary used to sound like a broken
record.
"""
from __future__ import annotations

import random
import re
from typing import Iterable, List, Optional

_TRAILING_PUNCT = re.compile(r"[\s.\"!]+$")
_WS = re.compile(r"\s+")


def normalize_saying(text: Optional[str]) -> Optional[str]:
    """Normalize a saying; returns None when it is not usable."""
    if not text or not text.strip():
        return None
    t = _WS.sub(" ", text.strip())
    if len(t) < 3 or len(t) > 300:
        return None
    trimmed = _TRAILING_PUNCT.sub("", t)
    return trimmed if len(trimmed) >= 3 else t


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}


class PhrasePool:
    """The mind's trained sayings, deduplicated and persisted in snapshots."""

    def __init__(self) -> None:
        self._sayings: List[str] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._sayings)

    @property
    def count(self) -> int:
        return len(self._sayings)

    def add(self, saying: str) -> bool:
        """Registers a saying; returns True when it was new."""
        clean = normalize_saying(saying)
        if clean is None:
            return False
        for existing in self._sayings:
            if existing.lower() == clean.lower():
                return False
        self._sayings.append(clean)
        return True

    def all(self) -> List[str]:
        return list(self._sayings)

    def remove(self, saying: str) -> bool:
        """Removes a saying (case-insensitive match). Used by the
        self-improvement module to retire sayings that consistently drew
        disapproval — the mind stops saying what never lands. Returns True
        when a saying was removed."""
        clean = normalize_saying(saying)
        if clean is None:
            return False
        for i, existing in enumerate(self._sayings):
            if existing.lower() == clean.lower():
                self._sayings.pop(i)
                return True
        return False

    def restore(self, sayings: Iterable[str]) -> None:
        for s in sayings:
            self.add(s)

    def pick_relevant(self, context_text: str, rng: random.Random, max_items: int) -> List[str]:
        """Up to max_items sayings, best first. Relevance is word overlap
        with the conversation text; when nothing overlaps, a rotating window
        keeps trained lines fresh instead of always surfacing the first one."""
        if max_items <= 0 or not self._sayings:
            return []
        context_words = _content_words(context_text or "")
        if not context_words:
            return self._rotated_sample(rng, max_items)

        scored = [(s, sum(1 for w in s.split() if w.lower() in context_words))
                  for s in self._sayings]
        scored.sort(key=lambda p: (-p[1], rng.random()))
        best = scored[0][1]
        if best == 0:
            return self._rotated_sample(rng, max_items)
        threshold = max(1, best // 2)
        pool = [s for s, score in scored if score >= threshold][:max_items]
        return pool or self._rotated_sample(rng, max_items)

    def _rotated_sample(self, rng: random.Random, max_items: int) -> List[str]:
        if not self._sayings:
            return []
        start = (self._cursor + rng.randrange(min(3, len(self._sayings)))) % len(self._sayings)
        self._cursor = (start + max_items) % len(self._sayings)
        n = min(max_items, len(self._sayings))
        return [self._sayings[(start + i) % len(self._sayings)] for i in range(n)]
