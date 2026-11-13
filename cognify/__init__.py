"""CognifyPy - a cognitive AI mind in pure Python.

Reasoning, thinking, self-learning, emotions - and vision, audio,
embeddings, multilingual, tool-use, code, and media generation.
"""
from .core.builder import MindBuilder
from .core.events import EventBus, MindEvent
from .core.gender import (FEMALE, MALE, UNKNOWN_PERSON, UNSPECIFIED,
                          GenderOptions, PronounSet)
from .core.mind import Mind, TrainingReport
from .core.options import ExpressionOptions, ThinkingCeiling, ThinkingLevel, ThinkingOptions
from .core.phrases import PhrasePool
from .core.stimulus import Modality, Stimulus
from .core.thought_stream import Thought, ThoughtStage, ThoughtStream
from .emotions.feeling import FeelingSnapshot

__version__ = "1.1.0"

__all__ = [
    "Mind", "MindBuilder", "Stimulus", "Modality",
    "EventBus", "MindEvent",
    "Thought", "ThoughtStage", "ThoughtStream",
    "PhrasePool", "TrainingReport",
    "GenderOptions", "PronounSet", "FEMALE", "MALE", "UNKNOWN_PERSON", "UNSPECIFIED",
    "ThinkingLevel", "ThinkingCeiling", "ThinkingOptions", "ExpressionOptions", "FeelingSnapshot",
    "__version__",
]
