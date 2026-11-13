"""HybridNextToken: the blend of the mind's two language models.

The statistical n-gram (:class:`~cognify.learning.next_token.NextTokenModel`)
is exact about what it has literally seen; the transformer
(:class:`~cognify.learning.transformer.TransformerLM`) generalizes to
contexts it never saw verbatim. Interpolation rule:
``p = w * p_neural + (1 - w) * p_ngram`` over the union of candidates,
renormalized over the returned top-k. An untrained transformer degrades
gracefully to pure n-gram.

Mirrors the C# ``Cognify.Learning.HybridNextToken``.
"""
from __future__ import annotations

from typing import List, Optional

from .next_token import NextTokenModel, TokenPrediction
from .transformer import TransformerLM


def predict_next(ngram: NextTokenModel, neural: Optional[TransformerLM],
                 context: str, count: int = 5,
                 neural_weight: float = 0.6) -> List[TokenPrediction]:
    """Blended next-token predictions after the context, best first."""
    count = max(1, count)
    from_ngram = ngram.predict_next(context, max(count * 5, 25))

    if neural is None or not neural.is_trained:
        return from_ngram[:count]

    from_neural = neural.predict_next(context, max(count * 5, 25))
    w = min(1.0, max(0.0, neural_weight))

    scores = {}
    for p in from_ngram:
        scores[p.token] = scores.get(p.token, 0.0) + (1.0 - w) * p.probability
    for p in from_neural:
        scores[p.token] = scores.get(p.token, 0.0) + w * p.probability

    top = sorted(scores.items(), key=lambda kv: -kv[1])[:count]
    total = sum(v for _, v in top)
    if total <= 0:
        return from_ngram[:count]
    return [TokenPrediction(tok, v / total) for tok, v in top]


def fluency(ngram: NextTokenModel, neural: Optional[TransformerLM],
            text: str, neural_weight: float = 0.5) -> float:
    """Blended fluency score — mean per-token log-probability under both
    models (higher, i.e. closer to 0 from below, reads more natural)."""
    ngram_score = ngram.score(text)
    if neural is None or not neural.is_trained:
        return ngram_score
    w = min(1.0, max(0.0, neural_weight))
    return w * neural.score(text) + (1.0 - w) * ngram_score
