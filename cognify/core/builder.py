"""MindBuilder: the fluent entry point for assembling a Mind."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..capabilities.base import Capability
from ..capabilities.code import CodeCapability
from ..capabilities.embeddings import EmbeddingService
from ..capabilities.media import MediaGenerationCapability
from ..capabilities.multilingual import MultilingualCapability
from ..capabilities.tool_use import ToolRegistry, ToolUseCapability, builtin_tools
from ..capabilities.vision import VisionCapability
from ..capabilities.audio import AudioCapability
from ..emotions.personality import PRESETS, Personality
from .mind import Mind
from .options import (ConversationalOptions, ExpressionOptions,
                      ThinkingCeiling, ThinkingLevel, ThinkingOptions)

#: Every toggleable capability name accepted by :meth:`without` / :meth:`only`.
KNOWN_CAPABILITIES: tuple = (
    "reasoning", "thinking", "learning", "emotions", "memory", "goals", "personality",
    "vision", "audio", "embeddings", "multilingual", "tool_use", "code", "media_generation",
)

#: Friendly aliases (Cognify-C# style names included).
_ALIASES: dict = {
    "tools": "tool_use",
    "tool_use": "tool_use",
    "generation": "media_generation",
    "media": "media_generation",
    "media_generation": "media_generation",
}


def _canonical(name: str) -> str:
    """Normalizes a capability name (case-insensitive, aliases resolved)."""
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in KNOWN_CAPABILITIES:
        raise ValueError(
            f"unknown capability {name!r}; known: {', '.join(KNOWN_CAPABILITIES)}")
    return key


class MindBuilder:
    """Fluent builder.

    Example::

        mind = (MindBuilder("Cognify")
                .with_preset("curious_child")
                .enable_vision()
                .enable_tool_use()
                .build())
    """

    def __init__(self, name: str = "Cognify") -> None:
        self.name = name
        self._seed = 42
        self._personality: Optional[Personality] = None
        self._gender: Optional["GenderOptions"] = None
        self._emotion_sensitivity = 1.0
        self._memory_capacity = 200
        self._encoder_dims = 128
        self._flags: Dict[str, bool] = {
            "reasoning": True, "thinking": True, "emotions": True,
            "memory": True, "goals": True, "learning": True,
        }
        self._caps: Dict[str, bool] = {
            "embeddings": True, "multilingual": False, "tool_use": False,
            "vision": False, "audio": False, "code": False,
            "media_generation": False,
        }
        self._tools: List[Callable] = []
        self._extra_caps: List[Capability] = []
        # Thinking levels + what the mind shows in chat (see options.py).
        self._thinking_options = ThinkingOptions()
        self._expression = ExpressionOptions()
        self._conversational = ConversationalOptions()
        self._allow_web_search = False
        self._auto_fetch_web = False
        self._web_search_backend = None
        self._explore_max_pages = 3

    # ------------------------------------------------------------------ core
    def with_name(self, name: str) -> "MindBuilder":
        self.name = name
        return self

    def with_seed(self, seed: int) -> "MindBuilder":
        self._seed = int(seed)
        return self

    def with_big_five(self, openness: float, conscientiousness: float,
                      extraversion: float, agreeableness: float,
                      neuroticism: float) -> "MindBuilder":
        self._personality = Personality(openness, conscientiousness, extraversion,
                                        agreeableness, neuroticism)
        return self

    def with_personality(self, personality: Personality) -> "MindBuilder":
        self._personality = personality
        return self

    def with_preset(self, preset: str) -> "MindBuilder":
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        self._personality = Personality.from_preset(preset)
        return self

    def with_random_personality(self, seed: Optional[int] = None) -> "MindBuilder":
        self._personality = Personality.random(seed)
        return self

    def with_emotion_sensitivity(self, value: float) -> "MindBuilder":
        self._emotion_sensitivity = max(0.1, min(3.0, float(value)))
        return self

    def with_memory_capacity(self, capacity: int) -> "MindBuilder":
        self._memory_capacity = max(10, int(capacity))
        return self

    def with_encoder_dims(self, dims: int) -> "MindBuilder":
        self._encoder_dims = max(16, int(dims))
        return self

    def with_curiosity(self, level: float) -> "MindBuilder":
        self._personality = self._personality or Personality()
        self._personality.openness = max(0.0, min(1.0, float(level)))
        return self

    # ------------------------------------------------------------------ gender
    def with_gender(self, identity: str,
                    pronouns: Optional["PronounSet"] = None) -> "MindBuilder":
        """Sets the mind's gender identity: how it describes itself and the
        pronouns others use for it.

        Examples::

            MindBuilder("Yume").with_gender("girl", FEMALE)
            MindBuilder("Sol").with_gender("he/him")          # pronouns only
            MindBuilder("Neo").with_gender("non-binary person", NON_BINARY)

        A mind without a gender never mentions one — behavior is unchanged.
        """
        from .gender import GenderOptions
        self._gender = GenderOptions().set(identity, pronouns)
        return self

    def with_pronouns(self, pronouns: "PronounSet") -> "MindBuilder":
        """Sets just the mind's pronoun set (no identity word)."""
        from .gender import GenderOptions, UNSPECIFIED
        self._gender = GenderOptions(identity=self._gender.identity
                                     if self._gender else "",
                                     pronouns=pronouns or UNSPECIFIED)
        return self

    # ------------------------------------------------------------------ modules
    def _set_flag(self, key: str, value: bool) -> "MindBuilder":
        self._flags[key] = bool(value)
        return self

    def with_reasoning(self, enabled: bool = True) -> "MindBuilder":
        return self._set_flag("reasoning", enabled)

    def with_thinking(self, enabled: bool = True,
                      level: "ThinkingLevel | str | None" = None,
                      tone: Optional[str] = None,
                      budget: Optional[int] = None,
                      ceiling: "ThinkingCeiling | str | None" = None,
                      automatic: Optional[bool] = None) -> "MindBuilder":
        """Thinking, with levels — the same idea as thinking-token budgets in
        modern language models.

        Examples::

            .with_thinking()                                  # default (medium, balanced)
            .with_thinking(False)                             # this model cannot think before speaking
            .with_thinking(level="high", tone="poetic")       # 4 thoughts, in a soft voice
            .with_thinking(level="max")                       # the full inner monologue (8 thoughts)
            .with_thinking(budget=300)                        # explicit token budget (~6 thoughts)
            .with_thinking(ceiling="only2")                   # this model may never think above Medium

        Levels: ``low`` (1 thought) / ``medium`` (2) / ``high`` (4) /
        ``veryhigh`` (6) / ``max`` (8). ``tone``: ``balanced``, ``focused``,
        ``playful``, ``poetic`` or ``plain``. ``ceiling`` — the hard cap
        (Only1 = Low only, Only2 = up to Medium, Only3 = up to High, Only4 =
        up to Very High, Only5 = everything, the default): any level
        requested at or above the cap — including runtime changes — is
        clamped down, and the cap is exported with the model
        (``cognify.thinking.ceiling``) so serving stacks can read it. The
        generated thoughts are produced BEFORE the reply — read them any
        time via ``mind.last_thoughts``.
        """
        self._set_flag("thinking", enabled)
        self._thinking_options.enable(enabled)
        if ceiling is not None:
            self._thinking_options.up_to(ceiling)
        if level is not None:
            self._thinking_options.at(level)
        if tone is not None:
            self._thinking_options.with_tone(tone)
        if budget is not None:
            self._thinking_options.budget(budget)
        if automatic is not None:
            # THINKING AUTOMATIC — she picks the effort per topic herself
            # (the configured level becomes the base; the ceiling still caps).
            self._thinking_options.automatic_on(automatic)
        return self

    def with_conversational(
            self,
            configure: Optional[Callable[[ConversationalOptions], None]] = None,
            rate: Optional[float] = None,
            understand: Optional[bool] = None) -> "MindBuilder":
        """Configures how TALKATIVE the mind is and how much it collects from
        the conversation::

            .with_conversational(rate=1.5, understand=True)
            .with_conversational(lambda c: c.rate_amount(2.0))

        ``rate`` scales reply length (1.0 default; 1.5+ adds conversational
        sentences drawn from related knowledge, memories and the conversation
        itself), ``understand`` keeps topic-shift notes so recall knows the
        SHAPE of the conversation, not just lines."""
        if rate is not None:
            self._conversational.rate_amount(rate)
        if understand is not None:
            self._conversational.understand_shifts(understand)
        if configure is not None:
            configure(self._conversational)
        return self

    # ------------------------------------------------------------------ web
    def with_web_search(self, backend: Optional[Any] = None) -> "MindBuilder":
        """Registers the MANUAL-ONLY web tools: web search never fires on its
        own inside the cognitive cycle — it runs only when explicitly
        requested via ``mind.search_web(...)`` or the demo's ``/searchweb``
        command. ``backend`` may be a custom WebSearchBackend."""
        self._flags["tools"] = True
        self._web_search_backend = backend
        self._allow_web_search = True
        return self

    def with_auto_web_fetch(self, enabled: bool = True,
                            backend: Optional[Any] = None,
                            explore_pages: int = 3) -> "MindBuilder":
        """Gives the mind AUTONOMOUS WEB LOOKUP — her own WebFetch. When a
        conversational topic is clearly outside her knowledge, she says so
        naturally ("I don't know much about X, but I'll search it and
        understand it a bit more"), looks it up herself, learns from the
        results, and answers from what she found. Registers the manual-only
        web tools too (so /searchweb and /explore keep working), but the auto
        path is the new ability: still guarded — one lookup per unfamiliar
        topic, no arithmetic/identity triggers, failed lookups stay silent.
        ``explore_pages``: how many pages one /explore site-walk reads
        (starting page + internal links, 1..6)."""
        self._flags["tools"] = enabled
        self._allow_web_search = True
        self._auto_fetch_web = enabled
        if backend is not None:
            self._web_search_backend = backend
        self._explore_max_pages = max(1, min(6, int(explore_pages)))
        return self

    def with_expression(self, configure: Callable[[ExpressionOptions], None]) -> "MindBuilder":
        """Controls what the mind SHOWS in chat — presentation, not cognition.

        Example::

            .with_expression(lambda e: e.feelings(True)      # announce feelings mechanically
                                     .follow_ups(False)      # never ask her own questions
                                     .thinking(True)         # hosts may show the thought trace
                                     .meta(True))            # hosts may show the meta line

        Defaults keep chat natural: feelings color the words (read the raw
        state any time via ``mind.feeling.current_status``), follow-up
        questions are on, internals stay hidden.
        """
        configure(self._expression)
        return self

    def with_emotions(self, enabled: bool = True) -> "MindBuilder":
        return self._set_flag("emotions", enabled)

    def with_memory(self, enabled: bool = True) -> "MindBuilder":
        return self._set_flag("memory", enabled)

    def with_goals(self, enabled: bool = True) -> "MindBuilder":
        return self._set_flag("goals", enabled)

    def with_learning(self, enabled: bool = True) -> "MindBuilder":
        return self._set_flag("learning", enabled)

    # ------------------------------------------------------------------ capabilities
    def _set_cap(self, key: str, value: bool) -> "MindBuilder":
        self._caps[key] = bool(value)
        return self

    def enable_embeddings(self, enabled: bool = True) -> "MindBuilder":
        return self._set_cap("embeddings", enabled)

    def enable_multilingual(self, enabled: bool = True) -> "MindBuilder":
        return self._set_cap("multilingual", enabled)

    def enable_vision(self, enabled: bool = True) -> "MindBuilder":
        return self._set_cap("vision", enabled)

    def enable_audio(self, enabled: bool = True) -> "MindBuilder":
        return self._set_cap("audio", enabled)

    def enable_code(self, enabled: bool = True) -> "MindBuilder":
        return self._set_cap("code", enabled)

    def enable_media_generation(self, enabled: bool = True) -> "MindBuilder":
        return self._set_cap("media_generation", enabled)

    def enable_tool_use(self, enabled: bool = True,
                        include_builtins: bool = True) -> "MindBuilder":
        self._set_cap("tool_use", enabled)
        if enabled and include_builtins and not self._tools:
            self._tools = builtin_tools()
        return self

    def with_tools(self, *funcs: Callable) -> "MindBuilder":
        """Enable tool use with the built-in set (clock, calculator, ...);
        any custom funcs are registered on top. Calling it with no arguments
        used to be a silent no-op — now it enables the capability, matching
        the C# ``WithTools()`` behavior."""
        self._tools.extend(funcs)
        if not self._tools:
            self._tools = builtin_tools()
        self._caps["tool_use"] = True
        return self

    def with_capability(self, capability: Capability) -> "MindBuilder":
        self._extra_caps.append(capability)
        return self

    # ------------------------------------------------------------------ batch toggles
    def without(self, *names: str) -> "MindBuilder":
        """Turns capabilities **off** in one call (case-insensitive, chainable).

        Accepts both core-module and capability names, e.g.::

            mind = (MindBuilder("Yume")
                    .without("audio", "tools", "code", "media_generation")
                    .build())

        ``"personality"`` resets the Big Five traits to neutral. Aliases:
        ``tools`` -> ``tool_use``, ``generation``/``media`` -> ``media_generation``.
        Unknown names raise :class:`ValueError`.
        """
        if not names:
            raise ValueError("without() needs at least one capability name; "
                             f"known: {', '.join(KNOWN_CAPABILITIES)}")
        for raw in names:
            key = _canonical(raw)
            if key == "personality":
                self._personality = Personality()          # neutral Big Five
                self._flags["personality"] = False
            elif key in self._flags:
                self._flags[key] = False
            else:
                self._caps[key] = False
                if key == "tool_use":
                    self._tools.clear()
        return self

    def only(self, *names: str) -> "MindBuilder":
        """Opt-in mode: turns **everything** off, then enables exactly the
        listed capabilities. This is the Python equivalent of the C# builder's
        opt-in style::

            mind = (MindBuilder("ExampleAI")
                    .only("thinking", "learning", "memory")
                    .build())          # exactly 3 modules, nothing else
        """
        if not names:
            raise ValueError("only() needs at least one capability name; "
                             f"known: {', '.join(KNOWN_CAPABILITIES)}")
        wanted = [_canonical(raw) for raw in names]
        for key in KNOWN_CAPABILITIES:
            if key == "personality":
                continue
            if key in self._flags:
                self._flags[key] = False
            else:
                self._caps[key] = False
        self._tools.clear()
        self._caps["embeddings"] = False
        for key in wanted:
            if key == "personality":
                continue
            if key in self._flags:
                self._flags[key] = True
            else:
                self._caps[key] = True
                if key == "tool_use":
                    self._tools = builtin_tools()
        return self

    def with_all_modules(self) -> "MindBuilder":
        """Enables every core module and capability with default options."""
        for key in self._flags:
            self._flags[key] = True
        for key in self._caps:
            self._caps[key] = True
        if not self._tools:
            self._tools = builtin_tools()
        return self

    # ------------------------------------------------------------------ build
    def build(self) -> Mind:
        from ..capabilities.tool_use import ToolRegistry

        capabilities: List[Capability] = []
        registry = ToolRegistry()
        if self._caps.get("embeddings"):
            capabilities.append(EmbeddingService(dims=self._encoder_dims))
        if self._caps.get("multilingual"):
            capabilities.append(MultilingualCapability())
        if self._caps.get("vision"):
            capabilities.append(VisionCapability())
        if self._caps.get("audio"):
            capabilities.append(AudioCapability())
        if self._caps.get("code"):
            capabilities.append(CodeCapability())
        if self._caps.get("media_generation"):
            capabilities.append(MediaGenerationCapability())
        if self._caps.get("tool_use"):
            for fn in self._tools:
                registry.register(fn)
            capabilities.append(ToolUseCapability(registry))
        capabilities.extend(self._extra_caps)

        personality = self._personality or Personality()
        return Mind(name=self.name, seed=self._seed, personality=personality,
                    emotion_sensitivity=self._emotion_sensitivity,
                    memory_capacity=self._memory_capacity,
                    encoder_dims=self._encoder_dims,
                    gender=self._gender,
                    thinking=self._thinking_options,
                    expression=self._expression,
                    conversational=self._conversational,
                    allow_web_search=self._allow_web_search,
                    auto_fetch_web=self._auto_fetch_web,
                    web_search_backend=self._web_search_backend,
                    explore_max_pages=self._explore_max_pages,
                    flags=dict(self._flags), capabilities=capabilities,
                    tools_registry=registry, user_tools=list(self._tools))

    def __repr__(self) -> str:  # pragma: no cover
        return f"MindBuilder(name={self.name!r})"
