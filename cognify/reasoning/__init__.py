"""Facts, knowledge base, and inference."""
from .facts import Fact, FactParser
from .inference import InferenceEngine
from .knowledge import KnowledgeBase

__all__ = ["Fact", "FactParser", "KnowledgeBase", "InferenceEngine"]
