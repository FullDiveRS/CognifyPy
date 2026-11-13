"""Deliberate thinking."""
from .chain_of_thought import ChainOfThought
from .thought_generator import generate as generate_thoughts

__all__ = ["ChainOfThought", "generate_thoughts"]
