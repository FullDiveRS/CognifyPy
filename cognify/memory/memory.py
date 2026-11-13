"""Associative memory with spreading-activation recall."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

from ..util import clamp, jaccard, now, tokenize

_ids = itertools.count(1)


@dataclass
class MemoryRecord:
    content: str
    kind: str = "episode"            # episode | fact | semantic
    importance: float = 0.5
    strength: float = 0.0
    created: float = field(default_factory=now)
    last_access: float = field(default_factory=now)
    tags: Set[str] = field(default_factory=set)
    links: Set[int] = field(default_factory=set)
    id: int = field(default_factory=lambda: next(_ids))

    def __post_init__(self) -> None:
        self.importance = clamp(float(self.importance))
        self.kind = self.kind if self.kind in ("episode", "fact", "semantic") else "episode"


class AssociativeMemory:
    """Episodic + semantic memory.

    Records are linked when they are similar (token overlap or an optional
    embedding-based similarity function). Recall spreads activation one hop
    through links, so related-but-not-literal memories can surface.
    """

    def __init__(self, capacity: int = 200,
                 similarity: Optional[Callable[[str, str], float]] = None) -> None:
        self.capacity = max(10, int(capacity))
        self.similarity = similarity
        self._records: List[MemoryRecord] = []

    # -- writing ---------------------------------------------------------------
    def add(self, content: str, kind: str = "episode", importance: float = 0.5,
            tags: Optional[Set[str]] = None) -> MemoryRecord:
        content = (content or "").strip()
        if not content:
            raise ValueError("cannot store empty memory")
        rec = MemoryRecord(content=content, kind=kind, importance=importance,
                           strength=clamp(0.4 + 0.4 * importance),
                           tags=set(tags or set()))
        self._link(rec)
        self._records.append(rec)
        if len(self._records) > self.capacity:
            self._evict()
        return rec

    def _link(self, rec: MemoryRecord) -> None:
        for other in self._records:
            sim = self._similarity(rec.content, other.content)
            if sim >= 0.30 or (rec.tags & other.tags):
                rec.links.add(other.id)
                other.links.add(rec.id)

    def _similarity(self, a: str, b: str) -> float:
        base = jaccard(tokenize(a), tokenize(b))
        if self.similarity is not None:
            return max(base, self.similarity(a, b))
        return base

    def _evict(self) -> None:
        weakest = min(self._records,
                      key=lambda r: (r.strength * (0.5 + 0.5 * r.importance)))
        self._records.remove(weakest)
        for r in self._records:
            r.links.discard(weakest.id)

    # -- reading ---------------------------------------------------------------
    def recall(self, query: str, k: int = 3) -> List[Tuple[MemoryRecord, float]]:
        """Top-k records by spreading activation."""
        if not self._records:
            return []
        qt = set(tokenize(query))
        scored: List[Tuple[MemoryRecord, float]] = []
        for rec in self._records:
            direct = jaccard(qt, tokenize(rec.content))
            if direct <= 0.0:
                continue
            act = direct * (0.4 + 0.6 * rec.strength) * (0.5 + 0.5 * rec.importance)
            scored.append((rec, act))
        # spread one hop from directly activated records
        seeded = {r.id for r, _ in scored}
        by_id = {r.id: r for r in self._records}
        for rec, act in list(scored):
            for lid in rec.links:
                if lid in seeded:
                    continue
                other = by_id.get(lid)
                if other is not None:
                    scored.append((other, act * 0.45))
            seeded.update(rec.links)
        scored.sort(key=lambda p: p[1], reverse=True)
        top = [(r, a) for r, a in scored[:k] if a > 0.01]
        for rec, _ in top:
            self.touch(rec)
        return top

    def touch(self, rec: MemoryRecord) -> None:
        rec.strength = clamp(rec.strength + 0.1)
        rec.last_access = now()

    def find(self, content: str) -> Optional[MemoryRecord]:
        content = content.strip().lower()
        for rec in self._records:
            if rec.content.lower() == content:
                return rec
        return None

    # -- maintenance ------------------------------------------------------------
    def tick(self, decay: float = 0.015) -> None:
        """Passive consolidation: unused traces fade, never below 0.1."""
        for rec in self._records:
            rec.strength = max(0.1, rec.strength * (1.0 - decay))

    def all(self) -> List[MemoryRecord]:
        return list(self._records)

    def by_kind(self, kind: str) -> List[MemoryRecord]:
        return [r for r in self._records if r.kind == kind]

    def size(self) -> int:
        return len(self._records)

    def to_list(self) -> List[dict]:
        return [{"content": r.content, "kind": r.kind, "importance": r.importance,
                 "strength": r.strength, "tags": sorted(r.tags)} for r in self._records]

    def load(self, items: List[dict]) -> None:
        self._records.clear()
        for it in items:
            rec = MemoryRecord(content=it["content"], kind=it.get("kind", "episode"),
                               importance=it.get("importance", .5),
                               strength=it.get("strength", .5),
                               tags=set(it.get("tags", [])))
            self._records.append(rec)
        for rec in self._records:
            self._link(rec)
