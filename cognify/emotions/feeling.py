"""FeelingSnapshot: the developer-facing read of how the mind feels RIGHT NOW.

The programmatic counterpart of what chat no longer announces — chat stays
natural by default, and the developer reads the state here instead::

    print(yume.feeling.current_status)
    # "feeling anticipation (intensity 0.15) - mood positive (+0.00)"

Mirrors the C# ``FeelingSnapshot`` / ``yume.Feeling.CurrentStatus``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .emotion_state import EmotionalState

__all__ = ["FeelingSnapshot"]


@dataclass
class FeelingSnapshot:
    """A one-line-readable snapshot of the mind's current emotional state."""

    name: str = "steady"          # humanized dominant emotion ("anticipation")
    intensity: float = 0.0        # intensity of the dominant emotion (0..1)
    valence: float = 0.0          # mood valence (-1 gloomy .. +1 bright)
    arousal: float = 0.0          # arousal level (0 calm .. 1 keyed up)
    mood: str = "neutral"         # "positive", "neutral" or "negative"
    current_status: str = "feeling steady"

    @classmethod
    def from_state(cls, state: EmotionalState) -> "FeelingSnapshot":
        dominant = state.dominant(threshold=0.05)
        intensity = state.get(dominant) if dominant else 0.0
        valence = state.valence
        if valence < -0.15:
            mood = "negative"
        elif valence > 0.15:
            mood = "positive"
        else:
            mood = "neutral"
        name = "steady" if intensity < 0.05 else str(dominant).lower()
        sign = f"{valence:+.2f}"
        if intensity < 0.05:
            status = f"feeling steady - mood {mood} ({sign})"
        else:
            status = (f"feeling {name} (intensity {intensity:.2f}) - "
                      f"mood {mood} ({sign})")
        return cls(name=name,
                   intensity=round(intensity, 3),
                   valence=round(valence, 3),
                   arousal=round(state.arousal, 3),
                   mood=mood,
                   current_status=status)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.current_status
