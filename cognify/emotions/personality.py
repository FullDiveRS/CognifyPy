"""Personality: the Big Five (OCEAN) model."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

from ..util import clamp

PRESETS: Dict[str, Dict[str, float]] = {
    "curious_child":    {"openness": .9, "conscientiousness": .35, "extraversion": .8, "agreeableness": .7, "neuroticism": .6},
    "calm_mentor":      {"openness": .7, "conscientiousness": .8, "extraversion": .5, "agreeableness": .85, "neuroticism": .2},
    "focused_analyst":  {"openness": .55, "conscientiousness": .9, "extraversion": .3, "agreeableness": .5, "neuroticism": .25},
    "resilient_optimist": {"openness": .75, "conscientiousness": .6, "extraversion": .85, "agreeableness": .7, "neuroticism": .15},
}


@dataclass
class Personality:
    """Big Five traits, each in [0, 1]. They modulate emotions and style."""

    openness: float = 0.5            # curiosity, love of novelty
    conscientiousness: float = 0.5   # discipline, order
    extraversion: float = 0.5        # sociability, expressiveness
    agreeableness: float = 0.5       # warmth, cooperation
    neuroticism: float = 0.5         # emotional reactivity

    def __post_init__(self) -> None:
        for f in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            setattr(self, f, clamp(float(getattr(self, f))))

    @staticmethod
    def from_preset(name: str) -> "Personality":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        return Personality(**PRESETS[name])

    @staticmethod
    def random(seed: int | None = None) -> "Personality":
        rng = random.Random(seed)
        return Personality(*[rng.uniform(0.15, 0.85) for _ in range(5)])

    # -- style hints used by the response composer ---------------------------
    @property
    def verbosity(self) -> float:
        return clamp(0.35 * self.extraversion + 0.35 * self.openness + 0.15)

    @property
    def warmth(self) -> float:
        return clamp(0.7 * self.agreeableness + 0.3 * (1.0 - self.neuroticism))

    def describe(self) -> str:
        def label(v: float) -> str:
            return "high" if v >= 0.65 else "low" if v <= 0.35 else "moderate"
        return (f"{label(self.openness)} openness, {label(self.conscientiousness)} "
                f"conscientiousness, {label(self.extraversion)} extraversion, "
                f"{label(self.agreeableness)} agreeableness, {label(self.neuroticism)} neuroticism")

    def to_dict(self) -> Dict[str, float]:
        return {"openness": self.openness, "conscientiousness": self.conscientiousness,
                "extraversion": self.extraversion, "agreeableness": self.agreeableness,
                "neuroticism": self.neuroticism}

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "Personality":
        return Personality(**{k: float(v) for k, v in data.items()})
