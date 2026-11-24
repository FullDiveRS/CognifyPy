"""The AUTOMATIC thinking resolver — mirrors the C# ``ThinkingAuto``.

Decides HOW MUCH to think about an individual topic or conversation turn —
the library's version of a reasoning model spending more or fewer reasoning
tokens depending on the question. Host-facing and deterministic on the
inputs (no randomness), and always clamped by the model's
:class:`~cognify.core.options.ThinkingCeiling`.

Relationship to the built-in cycle: when ``ThinkingOptions.automatic`` is
on, :class:`~cognify.core.mind.Mind` runs its own internal per-cycle
selection inside the cognitive loop. This module is the same IDEA as a
standalone, reusable resolver — hosts, demos and tools that reason outside
the loop call :func:`choose` directly.

The weights, in order of how much they usually matter:

* questions (+2 — thinking exists to answer them well),
* "why"/"how" probes (+1), long inputs (+1),
* novel ground — nothing known and nothing recalled (+1),
* a strong emotional surge (+1), a fresh epistemic gap (+1).
"""
from __future__ import annotations

from typing import Optional

from ..core.options import ThinkingCeiling, ThinkingLevel

_QUESTION_STARTS = ("what", "who", "when", "where", "why", "how",
                    "which", "can ", "does ", "do ", "is ", "are ")


def _starts_with_question_word(text: str) -> bool:
    first = text.lstrip()
    return any(first.startswith(w) for w in _QUESTION_STARTS)


def choose(text: Optional[str], has_direct_answer: bool, recalled_count: int,
           emotion_intensity: float, epistemic_gap: bool,
           ceiling: ThinkingCeiling) -> ThinkingLevel:
    """Chooses the thinking level for THIS turn from plain inputs.

    Deterministic on the inputs; returns one of ``ThinkingLevel.LOW..MAX``
    (never a flag), always clamped to ``ceiling``.
    """
    text = text or ""
    tier = 0

    # Questions deserve real deliberation — that is what thinking is for.
    question = "?" in text or _starts_with_question_word(text)
    if question:
        tier += 2
    if (" why " in text.lower() or " how " in text.lower()
            or text.lower().startswith("why ")
            or text.lower().startswith("how ")):
        tier += 1

    # Longer turns carry more to weigh.
    words = len([w for w in text.split(" ") if w])
    if words > 12:
        tier += 1

    # Novel ground: no direct knowledge and nothing recalled — the mind must
    # reason its way forward instead of reciting, so think more. Only for
    # turns that actually seek something: a bare "hi" stays a quick pause.
    if not has_direct_answer and recalled_count == 0 and ("?" in text or words > 3):
        tier += 1

    # Emotion: a strong surge deserves a pause before speaking.
    if emotion_intensity > 0.5:
        tier += 1

    # A fresh epistemic gap (the emotion module's curiosity flag).
    if epistemic_gap:
        tier += 1

    tier = max(0, min(4, tier))
    level = {
        0: ThinkingLevel.LOW,
        1: ThinkingLevel.MEDIUM,
        2: ThinkingLevel.HIGH,
        3: ThinkingLevel.VERY_HIGH,
    }.get(tier, ThinkingLevel.MAX)
    return ceiling.clamp(level)
