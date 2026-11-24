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

from dataclasses import dataclass, field
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

# The classic default thinking-token ladder per level: budget/50 gives the
# thought counts above (1/2/4/6/8). Developers override individual rungs via
# ``ThinkingOptions.budget(level, tokens)`` / ``mind.set_thinking_budget``;
# a global ``thought_budget`` overrides the whole ladder.
DEFAULT_LEVEL_BUDGETS: Dict[str, int] = {
    "low": 50, "medium": 100, "high": 200, "veryhigh": 300, "max": 400,
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


class ReasoningLevel(str, Enum):
    """How hard the mind reasons — the reasoning phase's OWN ladder,
    mirroring :class:`ThinkingLevel` rung for rung (``low`` … ``max``) so
    the two modules line up level for level while each keeps its own type.
    The level sizes the forward-chaining depth (:data:`DEPTH_FOR`) and the
    explanation-trace budget (:meth:`ReasoningOptions.tokens_for`)."""

    LOW = "low"                # a quick check — shallow chaining (2 passes)
    MEDIUM = "medium"          # the default: the historical depth (6 passes)
    HIGH = "high"              # a real deduction (8 passes)
    VERY_HIGH = "veryhigh"     # deep chaining (10 passes)
    MAX = "max"                # maximum chaining (12 passes)

    @classmethod
    def parse(cls, value: "ReasoningLevel | str | None") -> "ReasoningLevel":
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

    @classmethod
    def from_thinking(cls, level: "ThinkingLevel | str") -> "ReasoningLevel":
        """Maps a thinking level onto reasoning's own ladder (same rung —
        one cognitive level drives both modules)."""
        return cls.parse(ThinkingLevel.parse(level).value)

    def to_thinking(self) -> ThinkingLevel:
        """Maps onto the thinking ladder (same rung)."""
        return ThinkingLevel.parse(self.value)


class ReasoningCeiling(str, Enum):
    """How far the mind is ALLOWED to reason — the reasoning mirror of
    :class:`ThinkingCeiling`: a hard cap the model builder sets at build
    time, with the same five presets (``ONLY1`` = Low only … ``ONLY5`` =
    everything, the default). A level requested above the ceiling is
    silently clamped DOWN to the ceiling's top level, and the cap travels
    with the model into exported files (``cognify.reasoning.ceiling`` GGUF
    metadata) so serving stacks can read it too."""

    ONLY1 = "only1"    # Low only
    ONLY2 = "only2"    # Low, Medium
    ONLY3 = "only3"    # Low, Medium, High
    ONLY4 = "only4"    # Low, Medium, High, Very High
    ONLY5 = "only5"    # Low, Medium, High, Very High, Max — no cap

    @classmethod
    def parse(cls, value: "ReasoningCeiling | str | None") -> "ReasoningCeiling":
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

    def max_level(self) -> ReasoningLevel:
        """The highest :class:`ReasoningLevel` this ceiling allows."""
        return {
            ReasoningCeiling.ONLY1: ReasoningLevel.LOW,
            ReasoningCeiling.ONLY2: ReasoningLevel.MEDIUM,
            ReasoningCeiling.ONLY3: ReasoningLevel.HIGH,
            ReasoningCeiling.ONLY4: ReasoningLevel.VERY_HIGH,
        }.get(self, ReasoningLevel.MAX)

    def clamp(self, level: ReasoningLevel) -> ReasoningLevel:
        """Clamps a requested level down to the ceiling: at-or-below passes
        through untouched, anything above lands on the ceiling's top level."""
        top = self.max_level()
        order = list(ReasoningLevel)
        return level if order.index(level) <= order.index(top) else top

    @classmethod
    def from_thinking(cls, ceiling: "ThinkingCeiling | str") -> "ReasoningCeiling":
        """Maps a thinking ceiling onto reasoning's own ceiling (same
        preset — one cognitive-level call caps both modules)."""
        return cls.parse(ThinkingCeiling.parse(ceiling).value)


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

    # THINKING TOKEN BUDGETS PER LEVEL — how many tokens each level may
    # spend, the ladder from Low up to Max (or up to the ceiling's top
    # level: every read clamps the level first, and writes clamp too, so a
    # Max budget on an Only2 model lands on Medium). Defaults reproduce the
    # classic ladder (50/100/200/300/400). Keys are level VALUE strings
    # ("low".."max") so they serialize to JSON as-is. With ``automatic`` on,
    # the resolver only picks the LEVEL — the chosen level's budget then
    # sizes the thoughts, which is why a global ``thought_budget`` suppresses
    # the choice while these do not.
    level_budgets: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LEVEL_BUDGETS))

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

    def budget(self, tokens: int, level: "ThinkingLevel | str | None" = None) -> "ThinkingOptions":
        """Sets a thinking TOKEN budget. One argument: ONE global budget for
        every level (overrides the ladder and the automatic choice — the
        developer named the exact budget). Two arguments
        ``budget(tokens, level)``: the budget for ONE level — the ladder
        from Low up to Max, or up to the ceiling's top level when a clamp is
        set (the level is clamped at write time). ``tokens <= 0`` removes
        that level's custom entry, falling back to the default ladder.
        """
        if level is None:
            self.thought_budget = max(0, int(tokens))
            return self
        key = ThinkingLevel.parse(level).value
        clamped = self.ceiling.clamp(ThinkingLevel.parse(level)).value
        if int(tokens) <= 0:
            self.level_budgets.pop(clamped, None)
        else:
            self.level_budgets[clamped] = int(tokens)
        del key
        return self

    def tokens_for(self, level: "ThinkingLevel | str") -> int:
        """The thinking tokens the given level may spend right now: a global
        ``thought_budget`` wins; otherwise the level (clamped by the ceiling)
        is looked up in ``level_budgets``, falling back to the default
        ladder for removed entries."""
        if self.thought_budget > 0:
            return int(self.thought_budget)
        key = self.ceiling.clamp(ThinkingLevel.parse(level)).value
        tokens = self.level_budgets.get(key, 0)
        return int(tokens) if tokens > 0 else DEFAULT_LEVEL_BUDGETS[key]

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
        copied.level_budgets = dict(self.level_budgets)
        return copied.at(level)

    @property
    def thought_rounds(self) -> int:
        """Generated-thought count for the current level/budget (1..10)."""
        if self.thought_budget > 0:
            return max(1, min(10, int(self.thought_budget) // 50))
        return max(1, min(10, self.tokens_for(self.level) // 50))

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "level": ThinkingLevel.parse(self.level).value,
                "ceiling": ThinkingCeiling.parse(self.ceiling).value,
                "tone": self.tone, "thought_budget": int(self.thought_budget),
                "level_budgets": dict(self.level_budgets or {}),
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
        if data.get("level_budgets"):
            for key, tokens in dict(data["level_budgets"]).items():
                norm = str(key).lower().replace("_", "").replace(" ", "")
                if norm in DEFAULT_LEVEL_BUDGETS and int(tokens) > 0:
                    opts.level_budgets[norm] = int(tokens)
        if "automatic" in data:
            opts.automatic = bool(data["automatic"])
        return opts


#: The forward-chaining depth a reasoning level buys: Low 2 … Medium 6 …
#: Max 12 inference passes. Medium keeps the historical default of 6, so
#: existing minds reason exactly as far as before; higher levels explore
#: deeper. Keyed by ReasoningLevel — reasoning's own ladder, rung for rung
#: identical to the thinking one.
DEPTH_FOR: Dict[ReasoningLevel, int] = {
    ReasoningLevel.LOW: 2,
    ReasoningLevel.MEDIUM: 6,
    ReasoningLevel.HIGH: 8,
    ReasoningLevel.VERY_HIGH: 10,
    ReasoningLevel.MAX: 12,
}


def effort_name(level: Any) -> str:
    """The level name llama.cpp's ``--reasoning-effort`` ladder uses:
    low / medium / high / xhigh / max. Accepts a ThinkingLevel or a
    ReasoningLevel (the ladders share values); Cognify's VeryHigh maps to
    ``xhigh`` so an exported level can be passed straight to the server."""
    value = getattr(level, "value", level)
    value = str(value).strip().lower()
    return {"veryhigh": "xhigh"}.get(value, value)


@dataclass
class ReasoningOptions:
    """Options for the reasoning phase — the knowledge/inference engine's
    settings, on reasoning's OWN ladder (:class:`ReasoningLevel`,
    :class:`ReasoningCeiling`) — rung for rung LINED UP WITH THINKING: the
    same five levels, the same ceiling presets and the same default budgets
    (50/100/200/300/400). One call (``MindBuilder.with_cognitive_level``)
    sets the effort of both modules; each can also be leveled
    independently. The level sizes the forward-chaining depth
    (:data:`DEPTH_FOR`) and the explanation-trace budget
    (:meth:`tokens_for`).
    """

    level: ReasoningLevel = ReasoningLevel.MEDIUM
    ceiling: ReasoningCeiling = ReasoningCeiling.ONLY5

    # The mind's own reasoning switch — can this model chain inferences at
    # all? False short-circuits the reasoning phase: facts already learned
    # stay queryable, but no new deduction / induction / abduction runs.
    # Mirrors the builder's with_reasoning(..., enabled=False) but stays
    # changeable at runtime — the GGUF exporter reads it for
    # cognify.reasoning.enabled, the same way it reads thinking's.
    enabled: bool = True

    # Explicit reasoning token budget overriding everything when > 0: the
    # token ceiling for the reasoning trace (learned-fact, arithmetic,
    # inference and induction explanations) in ONE cycle. One budget for
    # every level; for a per-level ladder use budget(tokens, level).
    token_budget: int = 0

    # Forward-chaining depth (inference passes). Sized by the level ladder
    # unless depth() pinned an explicit value.
    max_inference_depth: int = 6
    depth_is_custom: bool = False

    # Unsupervised induction of new facts from repeated co-occurrence, and
    # the observations required before a fact is induced (historical
    # default 3 on this side).
    allow_induction: bool = True
    induction_threshold: int = 3

    # Per-level trace budget ladder — same numbers as thinking by default.
    # Keys are level VALUE strings ("low".."max") so they serialize as-is.
    level_budgets: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LEVEL_BUDGETS))

    # -- fluent setters (chainable, like the C# versions) --------------------

    def enable(self, value: bool) -> "ReasoningOptions":
        """Flips the reasoning switch — the same switch thinking's
        ``enable()`` flips, on reasoning's own side."""
        self.enabled = bool(value)
        return self

    def depth(self, passes: int) -> "ReasoningOptions":
        """Pins an explicit forward-chaining depth (1..32) — the level
        ladder then stops retuning it. Serialized with the config so saved
        minds keep the pin."""
        self.max_inference_depth = max(1, min(32, int(passes)))
        self.depth_is_custom = True
        return self

    def up_to(self, ceiling: "ReasoningCeiling | str") -> "ReasoningOptions":
        """Caps how far this model may ever reason; the level is re-clamped
        immediately (mirror of ThinkingOptions.up_to)."""
        self.ceiling = ReasoningCeiling.parse(ceiling)
        self.level = self.ceiling.clamp(self.level)
        return self

    def at(self, level: "ReasoningLevel | str") -> "ReasoningOptions":
        """Sets the reasoning level — clamped by the ceiling. Re-sizes the
        inference depth from the level ladder unless depth() pinned one;
        the trace budget follows through tokens_for() at use time."""
        self.level = self.ceiling.clamp(ReasoningLevel.parse(level))
        if not self.depth_is_custom:
            self.max_inference_depth = DEPTH_FOR[self.level]
        return self

    def budget(self, tokens: int,
               level: "ReasoningLevel | str | None" = None) -> "ReasoningOptions":
        """Sets a reasoning TOKEN budget. One argument: ONE global budget
        for every level (overrides the ladder). Two arguments
        ``budget(tokens, level)``: the budget for ONE level — the ladder
        from Low up to Max, or up to the ceiling's top level when a clamp
        is set (clamped at write time). ``tokens <= 0`` removes that
        level's custom entry, falling back to the default ladder."""
        if level is None:
            self.token_budget = max(0, int(tokens))
            return self
        clamped = self.ceiling.clamp(ReasoningLevel.parse(level)).value
        if int(tokens) <= 0:
            self.level_budgets.pop(clamped, None)
        else:
            self.level_budgets[clamped] = int(tokens)
        return self

    def tokens_for(self, level: "ReasoningLevel | str") -> int:
        """The reasoning tokens the given level may spend on its trace: a
        global ``token_budget`` wins; otherwise the level (clamped by the
        ceiling) is looked up in ``level_budgets``, falling back to the
        default ladder. Same resolution rules as
        ThinkingOptions.tokens_for, so the two modules line up level for
        level."""
        if self.token_budget > 0:
            return int(self.token_budget)
        key = self.ceiling.clamp(ReasoningLevel.parse(level)).value
        tokens = self.level_budgets.get(key, 0)
        return int(tokens) if tokens > 0 else DEFAULT_LEVEL_BUDGETS[key]

    @property
    def effective_depth(self) -> int:
        """The depth the next cycle will chain: an explicit depth() wins,
        otherwise the level ladder (DEPTH_FOR) sizes it."""
        if self.depth_is_custom:
            return int(self.max_inference_depth)
        return DEPTH_FOR[ReasoningLevel.parse(self.level)]

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": bool(self.enabled),
                "level": ReasoningLevel.parse(self.level).value,
                "ceiling": ReasoningCeiling.parse(self.ceiling).value,
                "token_budget": int(self.token_budget),
                "max_inference_depth": int(self.max_inference_depth),
                "depth_is_custom": bool(self.depth_is_custom),
                "allow_induction": bool(self.allow_induction),
                "induction_threshold": int(self.induction_threshold),
                "level_budgets": dict(self.level_budgets or {})}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReasoningOptions":
        data = data or {}
        opts = cls()
        if "enabled" in data:
            opts.enabled = bool(data["enabled"])
        if data.get("ceiling"):
            opts.ceiling = ReasoningCeiling.parse(data["ceiling"])
        if data.get("level"):
            opts.level = opts.ceiling.clamp(ReasoningLevel.parse(data["level"]))
        if data.get("token_budget"):
            opts.token_budget = max(0, int(data["token_budget"]))
        if data.get("max_inference_depth"):
            opts.max_inference_depth = max(1, min(32, int(data["max_inference_depth"])))
        if "depth_is_custom" in data:
            opts.depth_is_custom = bool(data["depth_is_custom"])
        if "allow_induction" in data:
            opts.allow_induction = bool(data["allow_induction"])
        if data.get("induction_threshold"):
            opts.induction_threshold = int(data["induction_threshold"])
        if data.get("level_budgets"):
            for key, tokens in dict(data["level_budgets"]).items():
                norm = str(key).lower().replace("_", "").replace(" ", "")
                if norm in DEFAULT_LEVEL_BUDGETS and int(tokens) > 0:
                    opts.level_budgets[norm] = int(tokens)
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
