"""Tests for the thinking ceiling — the hard cap on how far a model may
think (Only1 = Low only … Only5 = everything). Mirrors the C# ceiling tests
in ThinkingTests."""
import pytest

from cognify import MindBuilder
from cognify.core.options import ThinkingCeiling, ThinkingLevel, ThinkingOptions


def make_mind(**thinking_kwargs):
    builder = (MindBuilder("CappedThinker")
               .with_reasoning()
               .with_thinking()
               .with_memory()
               .with_emotions())
    if thinking_kwargs:
        builder.with_thinking(**thinking_kwargs)
    return builder.build()


class TestCeilingLadder:
    def test_five_presets_map_to_their_top_level(self):
        assert ThinkingCeiling.ONLY1.max_level() == ThinkingLevel.LOW
        assert ThinkingCeiling.ONLY2.max_level() == ThinkingLevel.MEDIUM
        assert ThinkingCeiling.ONLY3.max_level() == ThinkingLevel.HIGH
        assert ThinkingCeiling.ONLY4.max_level() == ThinkingLevel.VERY_HIGH
        assert ThinkingCeiling.ONLY5.max_level() == ThinkingLevel.MAX

    def test_parse_accepts_enums_numbers_and_words(self):
        assert ThinkingCeiling.parse("only3") == ThinkingCeiling.ONLY3
        assert ThinkingCeiling.parse("3") == ThinkingCeiling.ONLY3
        assert ThinkingCeiling.parse("Only-4") == ThinkingCeiling.ONLY4
        assert ThinkingCeiling.parse(ThinkingCeiling.ONLY2) == ThinkingCeiling.ONLY2
        assert ThinkingCeiling.parse("nonsense") == ThinkingCeiling.ONLY5  # default: no cap
        assert ThinkingCeiling.parse(None) == ThinkingCeiling.ONLY5


class TestClamping:
    def test_only1_clamps_every_requested_level_to_low(self):
        t = ThinkingOptions().at("max").up_to("only1")
        assert t.level == ThinkingLevel.LOW
        assert t.thought_rounds == 1
        t.at("high")  # later requests are clamped too
        assert t.level == ThinkingLevel.LOW
        assert t.ceiling.clamp(ThinkingLevel.MAX) == ThinkingLevel.LOW

    def test_only3_clamps_above_high_and_passes_below(self):
        t = ThinkingOptions().up_to("only3")
        t.at("veryhigh")
        assert t.level == ThinkingLevel.HIGH
        t.at("max")
        assert t.level == ThinkingLevel.HIGH
        assert t.thought_rounds == 4        # High's promise, not Max's 8
        t.at("medium")                       # below the ceiling — untouched
        assert t.level == ThinkingLevel.MEDIUM

    def test_only5_is_the_uncapped_default(self):
        t = ThinkingOptions().at("max")
        assert t.ceiling == ThinkingCeiling.ONLY5
        assert t.level == ThinkingLevel.MAX
        assert t.thought_rounds == 8

    def test_setting_ceiling_reclamps_current_level(self):
        t = ThinkingOptions().at("max").up_to("only2")
        assert t.level == ThinkingLevel.MEDIUM
        assert t.thought_rounds == 2


class TestBuilderPreset:
    def test_builder_ceiling_clamps_runtime_level_changes(self):
        mind = make_mind(ceiling="only2")   # Low or Medium, nothing above
        assert mind.thinking_options.ceiling == ThinkingCeiling.ONLY2

        mind.thinking_options.at("max")     # a runtime request for Max…
        assert mind.thinking_options.level == ThinkingLevel.MEDIUM  # …thinks at Medium

        reply = mind.process("hello")
        assert len(mind.last_thoughts) <= 2  # Medium's token-set promise

    def test_builder_level_and_ceiling_order_does_not_matter(self):
        a = make_mind(level="max", ceiling="only1")
        b = make_mind(ceiling="only1", level="max")
        assert a.thinking_options.level == ThinkingLevel.LOW
        assert b.thinking_options.level == ThinkingLevel.LOW

    def test_config_round_trip_keeps_the_ceiling(self):
        opts = ThinkingOptions().at("max").up_to("only3")
        restored = ThinkingOptions.from_dict(opts.to_dict())
        assert restored.ceiling == ThinkingCeiling.ONLY3
        assert restored.level == ThinkingLevel.HIGH
        assert restored.thought_rounds == 4
