"""HashingEncoder: deterministic feature hashing from text to dense vectors."""
from __future__ import annotations

import re
from typing import List, Sequence

from ..util import cosine, l2_normalize, stable_hash, tokenize

_TOKEN_RE = re.compile(r"[\w\u00c0-\u024f\u0400-\u04ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


class HashingEncoder:
    """The hashing trick: no vocabulary, no training, fully deterministic.

    Unigrams and bigrams are hashed into a fixed-size vector (sign trick to
    reduce collisions bias), then L2-normalized.  Uses CRC32 so embeddings
    are stable across processes and machines.
    """

    def __init__(self, dims: int = 128, use_bigrams: bool = True) -> None:
        if dims < 16:
            raise ValueError("embedding dims must be >= 16")
        self.dims = int(dims)
        self.use_bigrams = use_bigrams

    def _features(self, text: str) -> List[str]:
        tokens = _TOKEN_RE.findall((text or "").lower())
        feats = list(tokens)
        if self.use_bigrams:
            feats += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return feats

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dims
        for i, feat in enumerate(self._features(text)):
            h = stable_hash(feat)
            idx = h % self.dims
            sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
            weight = 1.0 if i < len(self._features(text)) and "_" not in feat else 0.7
            vec[idx] += sign * weight
        return l2_normalize(vec)

    def cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        return cosine(a, b)

    def similarity(self, text_a: str, text_b: str) -> float:
        return cosine(self.encode(text_a), self.encode(text_b))
