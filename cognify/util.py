"""Shared utilities for CognifyPy."""
from __future__ import annotations

import re
import time
import zlib
from typing import Iterable, List, Sequence

_WORD_RE = re.compile(r"[a-z\u00c0-\u024f\u0400-\u04ff']+")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into the closed interval [lo, hi]."""
    return lo if value < lo else hi if value > hi else value


def now() -> float:
    """Monotonic wall-clock timestamp."""
    return time.time()


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer that keeps latin/cyrillic/latin-ext words."""
    return _WORD_RE.findall((text or "").lower())


def stable_hash(text: str) -> int:
    """Deterministic 32-bit hash (unlike builtin hash which is salted)."""
    return zlib.crc32(text.encode("utf-8", "ignore")) & 0xFFFFFFFF


def sigmoid(x: float) -> float:
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    if x >= 0:
        return 1.0 / (1.0 + pow(2.718281828459045, -x))
    e = pow(2.718281828459045, x)
    return e / (1.0 + e)


def tanh(x: float) -> float:
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e = pow(2.718281828459045, 2 * x)
    return (e - 1) / (e + 1)


def l2_normalize(vec: List[float]) -> List[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return list(vec)
    return [v / norm for v in vec]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter)


def singularize(word: str) -> str:
    """Very light English plural reducer, good enough for fact parsing."""
    w = word.lower().strip()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ses"):
        return w[:-2]
    # "-ness" words (happiness, kindness) are already singular: stripping
    # the trailing 's' used to mangle them into "happines"/"kindnes".
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ness")):
        return w[:-1]
    return w
