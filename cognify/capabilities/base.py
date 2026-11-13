"""Capability contract and percept data structures.

This module is intentionally dependency-free from ``cognify.core`` so that
capabilities can be imported anywhere without circular imports.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.stimulus import Stimulus


@dataclass
class Percept:
    """Base result of a capability perceiving a stimulus."""

    modality: str = "generic"
    description: str = ""
    confidence: float = 0.5
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class VisionPercept(Percept):
    width: int = 0
    height: int = 0
    objects: List[Dict[str, Any]] = field(default_factory=list)
    palette: List[str] = field(default_factory=list)
    brightness: float = 0.5
    edge_density: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.modality = "vision"


@dataclass
class AudioPercept(Percept):
    duration: float = 0.0
    rms: float = 0.0
    dominant_hz: float = 0.0
    note: str = ""
    zcr: float = 0.0
    character: str = "unknown"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.modality = "audio"


@dataclass
class LanguagePercept(Percept):
    language: str = "unknown"
    script: str = "latin"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.modality = "language"


@dataclass
class CodePercept(Percept):
    kind: str = "analysis"          # analysis | generation
    language: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.modality = "code"


@dataclass
class MediaPercept(Percept):
    assets: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.modality = "media"


@dataclass
class ToolInvocation:
    name: str
    args: Dict[str, Any]
    result: Any = None
    success: bool = True
    error: str = ""

    def summary(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        call = f"{self.name}({args})"
        if self.success:
            return f"{call} -> {self.result}"
        return f"{call} failed: {self.error}"


class Capability(ABC):
    """A pluggable perception/action module of the mind."""

    name: str = "capability"
    description: str = ""
    phase: str = "perceive"        # "perceive" | "tools"
    enabled: bool = True

    @abstractmethod
    def can_handle(self, stimulus: Stimulus, ctx: Any = None) -> bool:
        """Return True if this capability wants to process the stimulus."""

    @abstractmethod
    def handle(self, stimulus: Stimulus, ctx: Any = None) -> Optional[Percept]:
        """Process the stimulus and return a Percept (or None)."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Capability {self.name} phase={self.phase}>"
