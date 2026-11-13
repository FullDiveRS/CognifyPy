"""NextTokenModel: the mind's n-gram language model — next-token prediction.

Real trigram language modeling with stupid backoff (Brants et al. 2007):
trigram count ratio, backed off to bigram (x0.4), then to a Laplace-smoothed
unigram (x0.16). The mind observes EVERYTHING it reads and says — user turns,
its own replies, taught facts, trained sayings, web search results — so its
predictions track the conversation it actually lives in.

Mirrors the C# Cognify.Learning.NextTokenModel exactly (same tokenization,
same backoff factors, same JSON payload shape).
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from ..util import tokenize as _util_tokenize

# Stupid-backoff factors (the classic 0.4, squared for two hops).
BACKOFF = 0.4
BACKOFF2 = 0.16

# Caps so a long-lived mind cannot grow forever.
_MAX_BIGRAM_KEYS = 120_000
_MAX_TRIGRAM_KEYS = 120_000
_SERIALIZE_CAP = 60_000

_SENTENCE_SPLIT = re.compile(r"[.!?\;\n\r]+")


class TokenPrediction:
    """One next-token prediction: the token and its normalized probability."""

    __slots__ = ("token", "probability")

    def __init__(self, token: str, probability: float) -> None:
        self.token = token
        self.probability = probability

    def __repr__(self) -> str:  # pragma: no cover
        return f"TokenPrediction({self.token!r}, {self.probability:.3f})"


class NextTokenModel:
    """Compact trigram model with stupid backoff."""

    def __init__(self) -> None:
        self._unigram: Dict[str, int] = {}
        self._bigram: Dict[str, Dict[str, int]] = {}
        self._trigram: Dict[str, Dict[str, int]] = {}  # key: "w1\x1w2"
        self._total_tokens = 0

    # ------------------------------------------------------------------ stats
    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def vocabulary(self) -> int:
        return len(self._unigram)

    @property
    def is_trained(self) -> bool:
        return self._total_tokens > 0

    # ------------------------------------------------------------------ learn
    def observe(self, text: Optional[str]) -> None:
        """Counts every 1/2/3-gram in every sentence of the text."""
        if not text or not text.strip():
            return
        for sentence in _SENTENCE_SPLIT.split(text):
            tokens = self._tokenize(sentence)
            for i, token in enumerate(tokens):
                self._unigram[token] = self._unigram.get(token, 0) + 1
                self._total_tokens += 1
                if i >= 1 and len(self._bigram) < _MAX_BIGRAM_KEYS:
                    followers = self._bigram.setdefault(tokens[i - 1], {})
                    followers[token] = followers.get(token, 0) + 1
                if i >= 2 and len(self._trigram) < _MAX_TRIGRAM_KEYS:
                    key = tokens[i - 2] + "\x01" + tokens[i - 1]
                    followers = self._trigram.setdefault(key, {})
                    followers[token] = followers.get(token, 0) + 1

    def merge(self, other: "NextTokenModel") -> None:
        """Merges another model's counts into this one (snapshot restore)."""
        if other is None:
            return
        for word, count in other._unigram.items():
            self._unigram[word] = self._unigram.get(word, 0) + count
        for w1, followers in other._bigram.items():
            mine = self._bigram.setdefault(w1, {})
            for w2, count in followers.items():
                mine[w2] = mine.get(w2, 0) + count
        for key, followers in other._trigram.items():
            mine = self._trigram.setdefault(key, {})
            for w3, count in followers.items():
                mine[w3] = mine.get(w3, 0) + count
        self._total_tokens += other._total_tokens

    # ------------------------------------------------------------------ model
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Same normalization family as the util tokenizer (lowercase words)."""
        if not text:
            return []
        return _util_tokenize(text)

    def _probability(self, w1: Optional[str], w2: Optional[str], next_token: str) -> float:
        """Stupid-backoff probability P(next | w1, w2)."""
        if w1 is not None and w2 is not None:
            tri = self._trigram.get(w1 + "\x01" + w2)
            if tri is not None:
                denom = sum(tri.values())
                count = tri.get(next_token, 0)
                if denom > 0 and count > 0:
                    return count / denom
        if w2 is not None:
            bi = self._bigram.get(w2)
            if bi is not None:
                denom = sum(bi.values())
                count = bi.get(next_token, 0)
                if denom > 0 and count > 0:
                    return BACKOFF * count / denom
        uni = self._unigram.get(next_token, 0)
        if self._total_tokens > 0:
            return BACKOFF2 * (uni + 1.0) / (self._total_tokens + len(self._unigram) + 1.0)
        return BACKOFF2 / max(1, len(self._unigram) + 1)

    def predict_next(self, context: str, count: int = 5) -> List[TokenPrediction]:
        """Next-token predictions after the context, best first. There is
        ALWAYS an answer once the model has seen anything: trigram candidates
        back off to bigram, then to the most frequent unigrams."""
        count = max(1, count)
        tokens = self._tokenize(context or "")
        w1 = tokens[-2] if len(tokens) >= 2 else None
        w2 = tokens[-1] if len(tokens) >= 1 else None

        seen: List[Tuple[str, float]] = []
        seen_set = set()

        def add(candidates: Dict[str, int], factor: float) -> None:
            denom = sum(candidates.values())
            if denom <= 0:
                return
            for token, c in candidates.items():
                if token not in seen_set:
                    seen_set.add(token)
                    seen.append((token, factor * c / denom))

        if w1 is not None and w2 is not None:
            tri = self._trigram.get(w1 + "\x01" + w2)
            if tri:
                add(tri, 1.0)
        if w2 is not None:
            bi = self._bigram.get(w2)
            if bi:
                add(bi, BACKOFF)
        if len(seen) < count and self._total_tokens > 0:
            top = sorted(self._unigram.items(), key=lambda kv: -kv[1])[:40]
            add(dict(top), BACKOFF2)

        if not seen:
            return []

        scored = [(token, self._probability(w1, w2, token)) for token, _ in seen]
        scored = [(t, p) for t, p in scored if p > 0]
        scored.sort(key=lambda pair: -pair[1])
        total = sum(p for _, p in scored)
        if total <= 0:
            return []
        return [TokenPrediction(t, p / total) for t, p in scored[:count]]

    def score(self, text: str) -> float:
        """Mean per-token stupid-backoff log-probability — a fluency estimate.
        0 means no opinion (untrained); closer to 0 from below = fluent."""
        if not text or not text.strip() or self._total_tokens == 0:
            return 0.0
        total, n = 0.0, 0
        for sentence in _SENTENCE_SPLIT.split(text):
            tokens = self._tokenize(sentence)
            for i, token in enumerate(tokens):
                w1 = tokens[i - 2] if i >= 2 else None
                w2 = tokens[i - 1] if i >= 1 else None
                p = self._probability(w1, w2, token)
                total += math.log(max(p, 1e-8))
                n += 1
        return total / n if n else 0.0

    # ------------------------------------------------------------------ state
    def to_dict(self) -> dict:
        bi = [[w1, w2, c] for w1, followers in self._bigram.items()
              for w2, c in followers.items()]
        tri = [[key.split("\x01")[0], key.split("\x01")[1], w3, c]
               for key, followers in self._trigram.items()
               for w3, c in followers.items()]
        if len(bi) > _SERIALIZE_CAP:
            bi = [r for r in bi if r[2] >= 2][:_SERIALIZE_CAP]
        if len(tri) > _SERIALIZE_CAP:
            tri = [r for r in tri if r[3] >= 2][:_SERIALIZE_CAP]
        return {"uni": [[w, c] for w, c in self._unigram.items()],
                "bi": bi, "tri": tri, "total": self._total_tokens}

    def load_dict(self, data: dict) -> None:
        if not data:
            return
        for row in data.get("uni", []):
            if len(row) == 2:
                self._unigram[row[0]] = int(row[1])
        for row in data.get("bi", []):
            if len(row) == 3:
                self._bigram.setdefault(row[0], {})[row[1]] = int(row[2])
        for row in data.get("tri", []):
            if len(row) == 4:
                self._trigram.setdefault(row[0] + "\x01" + row[1], {})[row[2]] = int(row[3])
        self._total_tokens = int(data.get("total", 0))
