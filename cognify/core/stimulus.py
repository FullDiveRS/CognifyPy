"""Stimulus: anything that can enter the Mind."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class Stimulus:
    """A single input to the mind: text, raw image/audio bytes, or code."""

    text: str = ""
    modality: Modality = Modality.TEXT
    data: Optional[bytes] = None
    salience: float = 0.5
    source: str = "user"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text is None:
            self.text = ""
        if self.data is not None and not isinstance(self.data, (bytes, bytearray)):
            raise TypeError("Stimulus.data must be bytes or None")
        if self.modality in (Modality.IMAGE, Modality.AUDIO) and self.data is None:
            raise ValueError(f"modality {self.modality.value} requires data bytes")

    # -- convenience constructors -------------------------------------------
    @staticmethod
    def text(text: str, salience: float = 0.5, source: str = "user", **meta: Any) -> "Stimulus":
        return Stimulus(text=text, modality=Modality.TEXT, salience=salience,
                        source=source, meta=dict(meta))

    @staticmethod
    def image(data: bytes, text: str = "", **meta: Any) -> "Stimulus":
        return Stimulus(text=text, modality=Modality.IMAGE, data=bytes(data),
                        salience=0.7, meta=dict(meta))

    @staticmethod
    def audio(data: bytes, text: str = "", **meta: Any) -> "Stimulus":
        return Stimulus(text=text, modality=Modality.AUDIO, data=bytes(data),
                        salience=0.7, meta=dict(meta))

    @staticmethod
    def code(text: str, language: Optional[str] = None, **meta: Any) -> "Stimulus":
        m = dict(meta)
        if language:
            m["language"] = language
        return Stimulus(text=text, modality=Modality.CODE, salience=0.7, meta=m)

    @property
    def is_empty(self) -> bool:
        return not self.text and self.data is None
