"""Thoughts and the stream they flow in."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List


class ThoughtStage(Enum):
    PERCEIVE = "perceive"
    QUESTION = "question"
    HYPOTHESIZE = "hypothesize"
    EVIDENCE = "evidence"
    EVALUATE = "evaluate"
    CONCLUDE = "conclude"
    REFLECT = "reflect"
    # The thought generator's pre-speech self-talk (thinking levels Low..Max
    # size it; the tone flavors it). Produced BEFORE the reply is composed —
    # the reply is the result the mind arrives at after these thoughts.
    GENERATE = "generate"


STAGE_ORDER: List[ThoughtStage] = list(ThoughtStage)


@dataclass
class Thought:
    stage: ThoughtStage
    content: str
    confidence: float = 0.5

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def __str__(self) -> str:
        return f"[{self.stage.value:>11}] {self.content} ({self.confidence:.2f})"


class ThoughtStream:
    """Ordered collection of thoughts produced by one cognitive cycle."""

    def __init__(self) -> None:
        self._thoughts: List[Thought] = []

    def add(self, stage: ThoughtStage, content: str, confidence: float = 0.5) -> Thought:
        thought = Thought(stage=stage, content=content, confidence=confidence)
        self._thoughts.append(thought)
        return thought

    def by_stage(self, stage: ThoughtStage) -> List[Thought]:
        return [t for t in self._thoughts if t.stage == stage]

    @property
    def conclusion(self) -> Thought | None:
        found = self.by_stage(ThoughtStage.CONCLUDE)
        return found[-1] if found else None

    @property
    def stages_visited(self) -> List[ThoughtStage]:
        return [t.stage for t in self._thoughts]

    def __len__(self) -> int:
        return len(self._thoughts)

    def __iter__(self) -> Iterator[Thought]:
        return iter(self._thoughts)

    def to_list(self) -> List[Thought]:
        return list(self._thoughts)

    def summary(self) -> str:
        return " -> ".join(t.stage.value for t in self._thoughts)
