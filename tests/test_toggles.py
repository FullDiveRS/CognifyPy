"""Tests for batch capability toggling: without(), only(), with_all_modules()."""
import pytest

from cognify import MindBuilder
from cognify.core.builder import KNOWN_CAPABILITIES


def cap_names(mind):
    return {c.name for c in mind.capabilities}


class TestOnly:
    def test_only_gives_exactly_the_requested_modules(self):
        mind = MindBuilder("ExampleAI").only("thinking", "learning", "memory").build()
        assert mind.flags["thinking"] is True
        assert mind.flags["learning"] is True
        assert mind.flags["memory"] is True
        assert mind.flags["reasoning"] is False
        assert mind.flags["emotions"] is False
        assert mind.flags["goals"] is False
        assert cap_names(mind) == set()          # no capabilities attached
        assert mind.learner is not None

    def test_only_is_case_insensitive_and_supports_aliases(self):
        mind = MindBuilder("X").only("THINKING", "Tools", "media").build()
        assert mind.flags["thinking"] is True
        assert cap_names(mind) == {"tool_use", "media_generation"}

    def test_only_minimal_mind_processes_stimuli(self):
        mind = MindBuilder("ExampleAI").only("thinking", "learning", "memory").build()
        reply = mind.process("hello world")
        assert reply.text
        assert mind.ask("how are you?")

    def test_only_accepts_learning_and_reinforces(self):
        mind = MindBuilder("X").only("learning").build()
        for text in ("one", "two", "three", "four"):
            mind.process(text)          # replay buffer fills at 4 -> net trains
        mind.feedback(0.7)
        mind.process("closing cycle")   # consumes the reward -> Q update
        assert mind.learner.net.samples_seen > 0
        assert mind.learner.agent.updates > 0

    def test_empty_only_raises(self):
        with pytest.raises(ValueError):
            MindBuilder("X").only()

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="smell"):
            MindBuilder("X").only("thinking", "smell")


class TestWithout:
    def test_without_strips_from_full_set(self):
        mind = (MindBuilder("Slim").with_all_modules()
                .without("audio", "tools", "generation", "code").build())
        names = cap_names(mind)
        assert "audio" not in names
        assert "tool_use" not in names
        assert "media_generation" not in names
        assert "code" not in names
        # everything else stays
        assert "vision" in names
        assert "embeddings" in names
        assert "multilingual" in names
        assert all(mind.flags[k] for k in ("reasoning", "thinking", "emotions",
                                           "memory", "goals", "learning"))

    def test_without_disables_core_flags(self):
        mind = MindBuilder("X").without("reasoning", "goals", "emotions").build()
        assert mind.flags["reasoning"] is False
        assert mind.flags["goals"] is False
        assert mind.flags["emotions"] is False
        assert mind.flags["thinking"] is True        # untouched

    def test_without_personality_resets_to_neutral(self):
        mind = MindBuilder("X").with_big_five(0.9, 0.9, 0.9, 0.9, 0.9) \
            .without("personality").build()
        assert mind.personality.openness == pytest.approx(0.5)
        assert mind.personality.neuroticism == pytest.approx(0.5)

    def test_without_is_chainable_and_case_insensitive(self):
        mind = MindBuilder("X").with_all_modules() \
            .without("AUDIO").without("Tools ").build()
        assert "audio" not in cap_names(mind)
        assert "tool_use" not in cap_names(mind)

    def test_without_unknown_raises(self):
        with pytest.raises(ValueError):
            MindBuilder("X").without("smell")

    def test_empty_without_raises(self):
        with pytest.raises(ValueError):
            MindBuilder("X").without()

    def test_without_keeps_mind_functional(self):
        mind = MindBuilder("Yume").with_all_modules() \
            .without("audio", "vision", "code", "media_generation").build()
        mind.teach("birds are animals")
        reply = mind.process("what is a bird?")
        assert reply.text


class TestWithAllModules:
    def test_with_all_modules_enables_everything(self):
        mind = MindBuilder("Full").with_all_modules().build()
        assert all(mind.flags.values())
        assert cap_names(mind) == {"embeddings", "multilingual", "vision",
                                   "audio", "code", "media_generation", "tool_use"}

    def test_known_capabilities_listing(self):
        assert "thinking" in KNOWN_CAPABILITIES
        assert "tool_use" in KNOWN_CAPABILITIES
        assert "media_generation" in KNOWN_CAPABILITIES
