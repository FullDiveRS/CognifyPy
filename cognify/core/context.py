"""CognitiveContext: everything one cycle of the mind knows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..capabilities.base import Percept, ToolInvocation
from ..emotions.emotion_state import EmotionalState
from ..emotions.personality import Personality
from ..memory.memory import MemoryRecord
from .stimulus import Stimulus
from .thought_stream import ThoughtStream

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from ..reasoning.facts import Fact


@dataclass
class ReasoningAnswer:
    """A structured answer produced by the inference engine."""

    text: str
    confidence: float
    kind: str = "answer"          # describe | yesno | why | causes | unknown
    chain: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.chain = tuple(self.chain)


@dataclass
class Response:
    """Final output of one cognitive cycle."""

    text: str = ""
    confidence: float = 0.5
    thoughts: List[Tuple[str, str, float]] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)
    learned_facts: List[str] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    assets: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveContext:
    """The working state shared by all modules during a single cycle."""

    cycle: int
    stimulus: Stimulus
    personality: Personality
    emotions: EmotionalState

    percepts: Dict[str, Percept] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    working_memory: List[MemoryRecord] = field(default_factory=list)
    recalled: List[Tuple[MemoryRecord, float]] = field(default_factory=list)
    learned_facts: List[Fact] = field(default_factory=list)
    derived_facts: List[Fact] = field(default_factory=list)
    answer: Optional[ReasoningAnswer] = None
    knowledge_size: int = 0
    active_goal: Optional[Any] = None
    thoughts: ThoughtStream = field(default_factory=ThoughtStream)
    tool_invocations: List[ToolInvocation] = field(default_factory=list)
    assets: List[Any] = field(default_factory=list)
    response: Optional[Response] = None
    scratch: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.stimulus.text or ""

    @property
    def has_question(self) -> bool:
        return "?" in self.text or self.text.lower().startswith(
            ("what", "why", "how", "can ", "is ", "does", "do ", "who", "when",
             "where", "define")
        )

    def percept_summary(self) -> str:
        parts = []
        for name, p in self.percepts.items():
            if not p.description or name in ("embeddings",):
                continue
            if getattr(p, "language", None) == "en":
                continue          # "it was English" carries no headline value
            parts.append(p.description)
        return " | ".join(parts)
