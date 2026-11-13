"""Developer-facing options: thinking levels and what the mind shows in chat.

Mirrors the C# ``ThinkingOptions`` / ``ExpressionOptions``:

* :class:`ThinkingLevel` — Low / Medium / High / Very High / Max, the same
  idea as thinking-token budgets in modern language models.
* :class:`ThinkingCeiling` — the hard cap on how far a model may think
  (Only1 = Low only … Only5 = everything); requests above it are clamped.
* :class:`ThinkingOptions` — the mind's own thinking switch (a boolean the
  developer controls), the level, the tone of the generated thoughts and an
  optional explicit token budget.
* :class:`ExpressionOptions` — what the mind SHOWS in conversation. Chat
  stays natural by default; internals are opt-in.
* :class:`ConversationalOptions` — how TALKATIVE the mind is (``rate``) and
  how much it collects from the conversation (``understand``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

__all__ = ["ThinkingLevel", "ThinkingCeiling", "ThinkingOptions",
           "ExpressionOptions", "ConversationalOptions"]


class ThinkingLevel(str, Enum):
    """How hard the mind thinks before it speaks — a larger level generates
    more internal thoughts (a bigger "thinking token set") before committing
    to a reply."""

    LOW = "low"                # a quick pause — one generated thought
    MEDIUM = "medium"          # the default: a brief think-through (two)
    HIGH = "high"              # a real deliberation (four)
    VERY_HIGH = "veryhigh"     # deep deliberation (six)
    MAX = "max"                # maximum deliberation (eight)

    @classmethod
    def parse(cls, value: "ThinkingLevel | str | None") -> "ThinkingLevel":
        """Accepts the enum, ``"veryhigh"``, ``"very_high"``, ``"very high"``
        (case-insensitive); unknown values fall back to MEDIUM."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            for member in cls:
                if member.value == key:
                    return member
        return cls.MEDIUM


#: Generated-thought count per level (the level's "token set" promise).
_LEVEL_ROUNDS: Dict[ThinkingLevel, int] = {
    ThinkingLevel.LOW: 1,
    ThinkingLevel.MEDIUM: 2,
    ThinkingLevel.HIGH: 4,
    ThinkingLevel.VERY_HIGH: 6,
    ThinkingLevel.MAX: 8,
}


class ThinkingCeiling(str, Enum):
    """How far the mind is ALLOWED to think — a hard cap the model builder
    sets at build time. Five presets, matching the ladder a model author
    picks from:

    * ``ONLY1`` — the mind may only think at Low.
    * ``ONLY2`` — Low or Medium.
    * ``ONLY3`` — Low, Medium or High.
    * ``ONLY4`` — Low, Medium, High or Very High.
    * ``ONLY5`` — everything up to Max (default).

    A level requested above the ceiling is silently clamped DOWN to the
    ceiling's top level — a model capped at Only3 asked for ``max`` still
    thinks, just at High. The cap travels with the model into exported
    files (``cognify.thinking.ceiling`` GGUF metadata) so serving stacks
    (llama.cpp / Unsloth) can read it too.
    """

    ONLY1 = "only1"    # Low only
    ONLY2 = "only2"    # Low, Medium
    ONLY3 = "only3"    # Low, Medium, High
    ONLY4 = "only4"    # Low, Medium, High, Very High
    ONLY5 = "only5"    # Low, Medium, High, Very High, Max — no cap

    @classmethod
    def parse(cls, value: "ThinkingCeiling | str | None") -> "ThinkingCeiling":
        """Accepts the enum, ``"only3"``, ``"3"`` (case-insensitive); unknown
        values fall back to ONLY5 (no cap)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            if key.startswith("only"):
                key = key[4:]
            if key.isdigit() and 1 <= int(key) <= 5:
                return cls(f"only{int(key)}")
            try:
                return cls(key)
            except ValueError:
                pass
        return cls.ONLY5

    def max_level(self) -> ThinkingLevel:
        """The highest :class:`ThinkingLevel` this ceiling allows."""
        return {
            ThinkingCeiling.ONLY1: ThinkingLevel.LOW,
            ThinkingCeiling.ONLY2: ThinkingLevel.MEDIUM,
            ThinkingCeiling.ONLY3: ThinkingLevel.HIGH,
            ThinkingCeiling.ONLY4: ThinkingLevel.VERY_HIGH,
        }.get(self, ThinkingLevel.MAX)

    def clamp(self, level: ThinkingLevel) -> ThinkingLevel:
        """Clamps a requested level down to the ceiling: at-or-below passes
        through untouched, anything above lands on the ceiling's top level."""
        top = self.max_level()
        order = list(ThinkingLevel)
        return level if order.index(level) <= order.index(top) else top


@dataclass
class ThinkingOptions:
    """Options for the thinking module — the thought generator's settings.

    ``enabled`` is the mind's own thinking switch: when False the model
    answers directly (from knowledge or a concise fallback) without the
    thought generator. ``level`` sizes the generated thought set.
    ``tone`` flavors HOW the thoughts read: ``"balanced"``, ``"focused"``,
    ``"playful"``, ``"poetic"`` or ``"plain"`` (unknown values fall back to
    balanced). ``thought_budget`` (tokens, > 0) overrides the level with
    roughly ``budget / 50`` generated thoughts (1..10).
    """

    enabled: bool = True
    level: ThinkingLevel = ThinkingLevel.MEDIUM
    ceiling: ThinkingCeiling = ThinkingCeiling.ONLY5
    tone: str = "balanced"
    thought_budget: int = 0

    # -- fluent setters (chainable, like the C# versions) --------------------
    def enable(self, value: bool) -> "ThinkingOptions":
        self.enabled = bool(value)
        return self

    def at(self, level: "ThinkingLevel | str") -> "ThinkingOptions":
        self.level = self.ceiling.clamp(ThinkingLevel.parse(level))
        return self

    def up_to(self, ceiling: "ThinkingCeiling | str") -> "ThinkingOptions":
        """Caps how far this model may ever think (Only1 = Low only … Only5
        = everything). The current level is re-clamped immediately and every
        later :meth:`at` request is clamped too."""
        self.ceiling = ThinkingCeiling.parse(ceiling)
        self.level = self.ceiling.clamp(ThinkingLevel.parse(self.level))
        return self

    def with_tone(self, tone: str) -> "ThinkingOptions":
        self.tone = (tone or "balanced").strip().lower() or "balanced"
        return self

    def budget(self, tokens: int) -> "ThinkingOptions":
        self.thought_budget = max(0, int(tokens))
        return self

    # THINKING AUTOMATIC — the mind chooses how much to think about each
    # topic or conversation itself. When True, ``level`` becomes the BASE
    # level and every cycle picks its own effort from the actual input: a
    # familiar simple line is answered with a quick Low think, while a long
    # "why/how" question or a fresh inference earns High..Max. The choice is
    # always clamped by ``ceiling`` and recorded in the chain of thought
    # ("Automatic: chose to think at High"), so it stays explainable. An
    # explicit ``thought_budget`` overrides the choice (the developer named
    # the exact token budget — that wins). Default False.
    automatic: bool = False

    def automatic_on(self, value: bool = True) -> "ThinkingOptions":
        self.automatic = bool(value)
        return self

    def for_level(self, level: "ThinkingLevel") -> "ThinkingOptions":
        """A copy of these options at ``level`` (ceiling still applies). Used
        by the automatic mode so the per-cycle choice never mutates the
        shared configuration object."""
        import copy
        copied = copy.copy(self)
        return copied.at(level)

    @property
    def thought_rounds(self) -> int:
        """Generated-thought count for the current level/budget (1..10)."""
        if self.thought_budget > 0:
            return max(1, min(10, int(self.thought_budget) // 50))
        return _LEVEL_ROUNDS[self.ceiling.clamp(ThinkingLevel.parse(self.level))]

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "level": ThinkingLevel.parse(self.level).value,
                "ceiling": ThinkingCeiling.parse(self.ceiling).value,
                "tone": self.tone, "thought_budget": int(self.thought_budget),
                "automatic": bool(self.automatic)}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ThinkingOptions":
        data = data or {}
        opts = cls()
        if "enabled" in data:
            opts.enabled = bool(data["enabled"])
        if data.get("ceiling"):
            opts.ceiling = ThinkingCeiling.parse(data["ceiling"])
        if data.get("level"):
            opts.level = opts.ceiling.clamp(ThinkingLevel.parse(data["level"]))
        if data.get("tone"):
            opts.tone = str(data["tone"])
        if data.get("thought_budget"):
            opts.thought_budget = max(0, int(data["thought_budget"]))
        if "automatic" in data:
            opts.automatic = bool(data["automatic"])
        return opts


@dataclass
class ExpressionOptions:
    """Controls what the mind SHOWS in conversation.

    Everything here is presentation, not cognition — the internals always
    run and stay available programmatically (``mind.feeling``,
    ``response.thoughts``, ``mind.last_thoughts``, ``response.meta``), but
    chat stays natural by default:

    * ``show_feeling_reads`` False (default): feelings color the words
      naturally instead of being announced. True brings back the mechanical
      introspection — "my current read: anticipation".
    * ``ask_follow_up_questions`` True (default): replies may end with the
      mind's own natural follow-up questions (the Pick(...) questions).
      False for a mind that only answers and never asks.
    * ``show_response_meta`` / ``show_thinking``: convention flags for chat
      hosts — the meta line and the thought trace stay on the response
      object either way; hosts surface them only when these are on.
    """

    show_feeling_reads: bool = False
    ask_follow_up_questions: bool = True
    show_response_meta: bool = False
    show_thinking: bool = False

    # -- fluent setters (chainable, like the C# versions) --------------------
    def feelings(self, value: bool) -> "ExpressionOptions":
        self.show_feeling_reads = bool(value)
        return self

    def follow_ups(self, value: bool) -> "ExpressionOptions":
        self.ask_follow_up_questions = bool(value)
        return self

    def meta(self, value: bool) -> "ExpressionOptions":
        self.show_response_meta = bool(value)
        return self

    def thinking(self, value: bool) -> "ExpressionOptions":
        self.show_thinking = bool(value)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "show_feeling_reads": self.show_feeling_reads,
            "ask_follow_up_questions": self.ask_follow_up_questions,
            "show_response_meta": self.show_response_meta,
            "show_thinking": self.show_thinking,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExpressionOptions":
        data = data or {}
        opts = cls()
        for key in ("show_feeling_reads", "ask_follow_up_questions",
                    "show_response_meta", "show_thinking"):
            if key in data:
                setattr(opts, key, bool(data[key]))
        return opts


@dataclass
class ConversationalOptions:
    """Conversational settings — how TALKATIVE the mind is and how much it
    collects from the conversation. Configured with
    ``with_conversational(lambda c: c.rate(amount))``.

    * ``rate`` — how much the mind speaks per reply, as a multiplier of its
      natural length. 1.0 (default) is the classic pacing; 1.5+ makes her
      chattier — replies grow an extra conversational sentence drawn from
      related knowledge, memories and the conversation itself — and 2.0+ may
      add a second one. Values below 1.0 keep replies shorter. Clamped to
      [0.5, 3.0].
    * ``understand`` — when True the mind tracks how the talk moves from
      topic to topic and keeps a compact understanding note in memory
      (source "conversation") whenever the subject shifts, so recall later
      knows not just isolated lines but the shape of the conversation.
    """

    rate: float = 1.0
    understand: bool = True

    # -- fluent setters (chainable, like the C# versions) --------------------
    def rate_amount(self, amount: float) -> "ConversationalOptions":
        """How talkative she is (0.5..3.0). Named ``rate_amount`` here — the
        C# side exposes the user-facing ``rate(amount)`` spelling."""
        self.rate = max(0.5, min(3.0, float(amount)))
        return self

    def understand_shifts(self, value: bool = True) -> "ConversationalOptions":
        self.understand = bool(value)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"rate": float(self.rate), "understand": bool(self.understand)}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConversationalOptions":
        data = data or {}
        opts = cls()
        if "rate" in data:
            opts.rate = max(0.5, min(3.0, float(data["rate"])))
        if "understand" in data:
            opts.understand = bool(data["understand"])
        return opts
