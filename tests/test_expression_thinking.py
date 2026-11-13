"""Tests for the developer-facing surface: the feeling API, the Expression
visibility switches, the optional follow-up questions, and the thinking
levels + thought generator. Mirrors the C# ExpressionAndThinkingTests."""
import pytest

from cognify import MindBuilder
from cognify.core.mind import Mind
from cognify.core.options import ExpressionOptions, ThinkingLevel, ThinkingOptions


def make_mind(thinking=None, expression=None):
    builder = (MindBuilder("Feel")
               .with_reasoning()
               .with_thinking()
               .with_memory()
               .with_emotions())
    if thinking is not None:
        builder.with_thinking(**thinking)
    if expression is not None:
        def configure(e, _kw=expression):
            for key, value in _kw.items():
                setattr(e, key, value)
        builder.with_expression(configure)
    return builder.build()


class TestFeelingApi:
    def test_current_status_available_after_a_cycle(self):
        mind = make_mind()
        mind.process("hello there")
        assert mind.feeling.current_status.startswith("feeling")
        assert mind.feeling.name
        assert 0.0 <= mind.feeling.intensity <= 1.0
        assert -1.0 <= mind.feeling.valence <= 1.0
        assert mind.feeling.mood in ("positive", "neutral", "negative")
        assert str(mind.feeling) == mind.feeling.current_status

    def test_default_chat_does_not_announce_the_read(self):
        mind = make_mind()
        reply = mind.process("I am learning so much today")
        # The mechanical introspection is opt-in now — feelings color the
        # words instead, and the state stays on mind.feeling.
        assert "my current read" not in reply.text

    def test_feeling_reads_flag_controls_mechanical_announcement(self):
        # Deterministic unit test of the emotional candidate renderer (the
        # Q-agent's style choice itself is probabilistic).
        from cognify.compose.composer import ResponseComposer
        from cognify.core.context import CognitiveContext
        from cognify.core.stimulus import Stimulus
        from cognify.emotions.emotion_state import EmotionalState
        from cognify.emotions.personality import Personality

        ctx = CognitiveContext(cycle=1, stimulus=Stimulus.text("I love this"),
                               personality=Personality(), emotions=EmotionalState())
        ctx.emotions.boost("joy", 0.5)

        loud = ResponseComposer(seed=1)
        loud.expression = ExpressionOptions(show_feeling_reads=True)
        out = loud._emotional_text(ctx, "This is lovely.")
        assert "my current read" in out
        assert "joy" in out

        quiet = ResponseComposer(seed=1)
        quiet.expression = ExpressionOptions()
        out = quiet._emotional_text(ctx, "This is lovely.")
        assert "my current read" not in out

    def test_default_never_announces_even_when_emotional_wins(self):
        quiet = make_mind()
        for text in ("I love this, you are wonderful!",
                     "I love this so much, thank you!",
                     "this makes me so happy, I love it!",
                     "what a joyful wonderful day, I love it"):
            reply = quiet.process(text)
            assert "my current read" not in reply.text

    def test_describe_self_shows_mood_only_when_feeling_reads_enabled(self):
        quiet = make_mind(expression={"show_feeling_reads": False})
        quiet.process("hello there")
        assert "Right now" not in quiet.describe_self()

        loud = make_mind(expression={"show_feeling_reads": True})
        loud.process("hello there")
        assert "Right now" in loud.describe_self()


class TestInternalsHiddenByDefault:
    def test_defaults(self):
        mind = make_mind()
        mind.process("hello")
        assert mind.expression.show_feeling_reads is False
        assert mind.expression.show_thinking is False
        assert mind.expression.show_response_meta is False
        assert mind.expression.ask_follow_up_questions is True

    def test_fluent_setters_chain(self):
        e = ExpressionOptions()
        assert e.feelings(True).follow_ups(False).meta(True).thinking(True) is e
        assert e.show_feeling_reads and e.show_thinking
        assert e.show_response_meta and not e.ask_follow_up_questions


class TestFollowUpsOptional:
    def test_disabled_never_asks_about_unknown_people(self):
        mind = MindBuilder("D").with_expression(
            lambda e: e.follow_ups(False)).build()
        for _ in range(6):
            reply = mind.process("I met someone new today")
            assert "Who are they to you" not in reply.text
            assert "What are they like" not in reply.text
            assert "Tell me more about them" not in reply.text

    def test_enabled_by_default(self):
        mind = make_mind()
        assert mind.expression.ask_follow_up_questions is True


class TestThinkingLevels:
    def test_levels_map_to_thought_rounds(self):
        opts = ThinkingOptions()
        assert opts.at("medium").thought_rounds == 2
        assert opts.at("low").thought_rounds == 1
        assert opts.at("high").thought_rounds == 4
        assert opts.at("veryhigh").thought_rounds == 6
        assert opts.at("max").thought_rounds == 8
        # an explicit token budget overrides the level (budget/50 thoughts)
        assert opts.budget(250).thought_rounds == 5

    def test_level_parsing_is_forgiving(self):
        assert ThinkingLevel.parse("VERY_HIGH") is ThinkingLevel.VERY_HIGH
        assert ThinkingLevel.parse("Very-High") is ThinkingLevel.VERY_HIGH
        assert ThinkingLevel.parse("nonsense") is ThinkingLevel.MEDIUM
        assert ThinkingLevel.parse(ThinkingLevel.MAX) is ThinkingLevel.MAX

    def test_generator_produces_thoughts_before_speaking(self):
        mind = MindBuilder("Y").with_thinking(level="high").build()
        reply = mind.process("hello there")
        # the level's promise is kept even on a bare first cycle
        assert len(mind.last_thoughts) == mind.thinking_options.thought_rounds == 4
        assert all(t for t in mind.last_thoughts)
        # the generated thoughts are part of the explainable chain too
        assert any(stage == "generate" for stage, _, _ in reply.thoughts)
        # ... and they come BEFORE the reply is composed (last_thoughts is
        # the generator's output for the cycle that produced this reply)
        assert reply.meta["action"] in {"direct", "emotional", "curious",
                                        "share_thought", "trained"}

    def test_disabled_thinking_short_circuits_the_generator(self):
        mind = MindBuilder("C").with_thinking(False).build()
        reply = mind.process("hello there")
        assert reply.text
        assert mind.last_thoughts == []
        assert not any(stage == "generate" for stage, _, _ in reply.thoughts)

    def test_thinking_tone_flavors_the_thoughts(self):
        mind = MindBuilder("T").with_thinking(tone="focused").build()
        mind.process("hello there")
        assert mind.last_thoughts
        assert all(t.startswith("Focus:") for t in mind.last_thoughts)

    def test_budget_overrides_level(self):
        mind = MindBuilder("B").with_thinking(budget=300).build()
        mind.process("hello there")
        assert len(mind.last_thoughts) == 6

    def test_builder_flag_and_options_stay_in_sync(self):
        mind = MindBuilder("S").with_thinking(False, level="max").build()
        assert mind.flags["thinking"] is False
        assert mind.thinking_options.enabled is False
        assert mind.thinking_options.level is ThinkingLevel.MAX


class TestPersistenceAndStats:
    def test_options_persist_through_snapshot(self):
        mind = MindBuilder("P").with_thinking(
            level="high", tone="poetic").with_expression(
            lambda e: e.feelings(True).follow_ups(False)).build()
        restored = Mind.from_dict(mind.to_dict())
        assert restored.thinking_options.level is ThinkingLevel.HIGH
        assert restored.thinking_options.tone == "poetic"
        assert restored.expression.show_feeling_reads is True
        assert restored.expression.ask_follow_up_questions is False
        # the composer picked the options back up (wiring survives restore)
        assert restored.composer.expression is restored.expression

    def test_stats_include_options(self):
        mind = make_mind()
        stats = mind.stats()
        assert stats["thinking"]["level"] == "medium"
        assert stats["expression"]["show_feeling_reads"] is False

    def test_options_to_dict_round_trip(self):
        t = ThinkingOptions().at("max").with_tone("plain").budget(400)
        t2 = ThinkingOptions.from_dict(t.to_dict())
        assert t2.level is ThinkingLevel.MAX
        assert t2.tone == "plain"
        assert t2.thought_budget == 400
        assert t2.thought_rounds == 8       # budget wins: 400 // 50 = 8
        e2 = ExpressionOptions.from_dict(
            {"show_thinking": True, "ask_follow_up_questions": False})
        assert e2.show_thinking and not e2.ask_follow_up_questions
