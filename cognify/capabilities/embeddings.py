"""Embeddings: deterministic vector embeddings + similarity service."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..core.stimulus import Stimulus
from ..learning.encoder import HashingEncoder
from ..util import cosine
from .base import Capability, Percept


class EmbeddingService(Capability):
    """Embeds every stimulus into a dense vector; feeds memory & learning.

    Also usable as a standalone service: ``svc.embed(text)``,
    ``svc.similarity(a, b)``, ``svc.nearest(query, items)``.
    """

    name = "embeddings"
    description = "Turns text into deterministic dense vectors for similarity."
    phase = "perceive"

    def __init__(self, dims: int = 128) -> None:
        self.encoder = HashingEncoder(dims=dims)

    # -- service API ----------------------------------------------------------
    @property
    def dims(self) -> int:
        return self.encoder.dims

    def embed(self, text: str) -> List[float]:
        return self.encoder.encode(text)

    def similarity(self, a: str, b: str) -> float:
        return self.encoder.similarity(a, b)

    def nearest(self, query: str, items: Sequence[Tuple[str, str]],
                k: int = 3) -> List[Tuple[str, float]]:
        """Rank (key, text) items by cosine similarity to the query."""
        if not items:
            return []
        qv = self.embed(query)
        scored = [(key, cosine(qv, self.embed(text))) for key, text in items]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    # -- capability API ---------------------------------------------------------
    def can_handle(self, stimulus: Stimulus, ctx=None) -> bool:
        return bool(stimulus.text)

    def handle(self, stimulus: Stimulus, ctx=None) -> Optional[Percept]:
        if ctx is not None:
            ctx.embedding = self.embed(stimulus.text)
        sim = self.similarity(stimulus.text, stimulus.text)
        return Percept(
            description=f"Embedded input into {self.dims}d vector space.",
            confidence=sim,                      # 1.0 by construction
            modality="embedding",
            attributes={"dims": self.dims},
        )
