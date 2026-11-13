"""KnowledgeBase: indexed fact store with inheritance-aware queries."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from .facts import Fact


class KnowledgeBase:
    """Stores facts and answers inheritance-aware structural queries."""

    def __init__(self) -> None:
        self._facts: List[Fact] = []
        self._index: Dict[Tuple[str, str, str], Fact] = {}
        self._by_subject: Dict[str, List[Fact]] = {}
        self._by_object: Dict[str, List[Fact]] = {}
        self.last_derivation: Tuple[str, ...] = ()

    # -- writing ---------------------------------------------------------------
    def add(self, fact: Fact) -> bool:
        """Add a fact; duplicate keys merge (keep max confidence). Returns new?"""
        existing = self._index.get(fact.key)
        if existing is not None:
            existing.confidence = max(existing.confidence, fact.confidence)
            if fact.chain and not existing.chain:
                existing.chain = fact.chain
            return False
        self._facts.append(fact)
        self._index[fact.key] = fact
        self._by_subject.setdefault(fact.subject, []).append(fact)
        self._by_object.setdefault(fact.object, []).append(fact)
        return True

    def add_many(self, facts) -> int:
        return sum(1 for f in facts if self.add(f))

    # -- reading ---------------------------------------------------------------
    def get(self, subject: str, predicate: str, obj: str) -> Optional[Fact]:
        return self._index.get((subject, predicate, obj))

    def about(self, subject: str, predicate: Optional[str] = None) -> List[Fact]:
        found = self._by_subject.get(subject, [])
        if predicate:
            found = [f for f in found if f.predicate == predicate]
        return list(found)

    def subjects(self) -> List[str]:
        return sorted(self._by_subject.keys())

    def facts(self) -> List[Fact]:
        return list(self._facts)

    def size(self) -> int:
        return len(self._facts)

    # -- class-hierarchy helpers -------------------------------------------------
    def parents(self, concept: str) -> List[str]:
        """Direct is-a parents of a concept."""
        return [f.object for f in self._by_subject.get(concept, [])
                if f.predicate == "is-a"]

    def ancestors(self, concept: str, max_depth: int = 6) -> List[Tuple[str, int]]:
        """All is-a ancestors with their depth (BFS, cycle-safe)."""
        seen: Dict[str, int] = {concept: 0}
        queue: deque = deque([(concept, 0)])
        out: List[Tuple[str, int]] = []
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for parent in self.parents(node):
                if parent not in seen:
                    seen[parent] = depth + 1
                    out.append((parent, depth + 1))
                    queue.append((parent, depth + 1))
        return out

    def has_isa_path(self, a: str, b: str) -> Optional[List[str]]:
        """Shortest is-a chain a -> ... -> b, or None."""
        if a == b:
            return [a]
        prev: Dict[str, str] = {a: ""}
        queue: deque = deque([a])
        while queue:
            node = queue.popleft()
            for parent in self.parents(node):
                if parent in prev:
                    continue
                prev[parent] = node
                if parent == b:
                    path = [b]
                    cur = node
                    while cur:
                        path.append(cur)
                        cur = prev[cur]
                    return list(reversed(path))
                queue.append(parent)
        return None

    # -- inheritance-aware property lookup ----------------------------------------
    def inherited(self, concept: str, predicate: str) -> List[Tuple[Fact, float]]:
        """Properties of a concept including those inherited from ancestors.

        Returns (fact, confidence_after_decay) sorted by confidence.
        """
        out: List[Tuple[Fact, float]] = []
        for f in self._by_subject.get(concept, []):
            if f.predicate == predicate and not f.is_negated:
                out.append((f, f.confidence))
        for ancestor, depth in self.ancestors(concept):
            decay = 0.9 ** depth
            for f in self._by_subject.get(ancestor, []):
                if f.predicate == predicate and not f.is_negated:
                    out.append((f, f.confidence * decay))
        out.sort(key=lambda p: p[1], reverse=True)
        return out

    def contradicts(self, concept: str, predicate: str, obj: str) -> Optional[Fact]:
        """Find an explicit negation that blocks a would-be fact."""
        neg_pred = "is-not-a" if predicate == "is-a" else "can-not" if predicate == "can" else None
        if neg_pred is None:
            return None
        return self._index.get((concept, neg_pred, obj))

    def to_list(self) -> List[dict]:
        return [f.to_dict() for f in self._facts]

    def load(self, items: List[dict]) -> None:
        for d in items:
            self.add(Fact.from_dict(d))
