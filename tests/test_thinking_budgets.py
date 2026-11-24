"""THINKING TOKEN BUDGETS PER LEVEL — the ladder from Low up to Max (or up
to the ceiling's top level when a clamp is set): the developer sets how many
tokens each level may spend, ``thought_rounds`` derives the generated thought
set (budget/50, 1..10), a global one-size budget overrides, and under
AUTOMATIC thinking the resolver only picks the level — the chosen level's
budget follows. Mirrors the C# ThinkingBudgetTests."""
import json

import pytest

from cognify import MindBuilder
from cognify.core.options import (DEFAULT_LEVEL_BUDGETS, ThinkingCeiling,
                                  ThinkingLevel, ThinkingOptions)


def make_mind(level=None, automatic=False, ceiling=ThinkingCeiling.ONLY5,
              budgets=()) -> object:
    """``budgets`` is an iterable of (tokens, level) applied through the
    public runtime API after the build."""
    builder = (MindBuilder("Budgeter")
               .with_reasoning()
               .with_memory()
               .with_emotions()
               .with_thinking(level=level, ceiling=ceiling,
                              automatic=automatic))
    mind = builder.build()
    for tokens, lvl in budgets:
        mind.set_thinking_budget(tokens, lvl)
    return mind


class TestLevelBudgets:
    def test_defaults_reproduce_the_classic_ladder(self):
        t = ThinkingOptions()
        assert t.tokens_for(ThinkingLevel.LOW) == 50
        assert t.tokens_for(ThinkingLevel.MEDIUM) == 100
        assert t.tokens_for(ThinkingLevel.HIGH) == 200
        assert t.tokens_for(ThinkingLevel.VERY_HIGH) == 300
        assert t.tokens_for(ThinkingLevel.MAX) == 400
        assert DEFAULT_LEVEL_BUDGETS == {"low": 50, "medium": 100, "high": 200,
                                         "veryhigh": 300, "max": 400}

    def test_per_level_budget_sizes_the_thought_set(self):
        t = ThinkingOptions().at(ThinkingLevel.MEDIUM)
        t.budget(250, ThinkingLevel.MEDIUM)
        assert t.tokens_for(ThinkingLevel.MEDIUM) == 250
        assert t.thought_rounds == 5                     # 250/50 — re-sized
        t.at(ThinkingLevel.LOW)
        assert t.thought_rounds == 1                     # Low untouched: 50
        assert t.tokens_for(ThinkingLevel.LOW) == 50

    def test_ceiling_clamps_budget_writes_and_reads(self):
        # With the clamp set, the ladder runs from Low up to the CLAMP
        # level: a Max budget on an Only2 model lands on Medium.
        t = ThinkingOptions().up_to(ThinkingCeiling.ONLY2)
        t.budget(600, ThinkingLevel.MAX)
        assert t.tokens_for(ThinkingLevel.MAX) == 600    # read clamps to Medium
        assert t.tokens_for(ThinkingLevel.MEDIUM) == 600
        assert t.level_budgets["max"] == 400             # default, unreachable
        assert ThinkingCeiling.ONLY2.max_level() == ThinkingLevel.MEDIUM

    def test_global_budget_overrides_every_per_level_entry(self):
        t = ThinkingOptions().budget(222, ThinkingLevel.HIGH).budget(350)
        assert t.tokens_for(ThinkingLevel.HIGH) == 350
        assert t.tokens_for(ThinkingLevel.LOW) == 350
        assert t.thought_rounds == 7

    def test_zero_budget_entry_falls_back_to_the_default(self):
        t = ThinkingOptions().budget(222, ThinkingLevel.HIGH)
        assert t.tokens_for(ThinkingLevel.HIGH) == 222
        t.budget(0, ThinkingLevel.HIGH)
        assert t.tokens_for(ThinkingLevel.HIGH) == 200   # default ladder again

    def test_for_level_copies_budgets_in_isolation(self):
        shared = ThinkingOptions().budget(222, ThinkingLevel.HIGH)
        per_cycle = shared.for_level(ThinkingLevel.MEDIUM)
        per_cycle.budget(999, ThinkingLevel.HIGH)
        assert shared.tokens_for(ThinkingLevel.HIGH) == 222   # untouched
        assert per_cycle.tokens_for(ThinkingLevel.HIGH) == 999

    def test_budgets_persist_through_serialization(self):
        mind = make_mind(level=ThinkingLevel.HIGH,
                         budgets=[(260, ThinkingLevel.HIGH),
                                  (640, ThinkingLevel.MAX)])
        data = json.loads(json.dumps(mind.thinking_options.to_dict()))
        restored = ThinkingOptions.from_dict(data)
        assert restored.tokens_for(ThinkingLevel.HIGH) == 260
        assert restored.tokens_for(ThinkingLevel.MAX) == 640
        assert restored.tokens_for(ThinkingLevel.LOW) == 50
        # aliased keys normalize ("very_high" -> "veryhigh")
        restored2 = ThinkingOptions.from_dict(
            {"level_budgets": {"very_high": 123}})
        assert restored2.tokens_for(ThinkingLevel.VERY_HIGH) == 123

    def test_mind_set_thinking_budget_changes_the_cycle(self):
        mind = make_mind(level=ThinkingLevel.LOW)
        mind.teach("Tulips are flowers. Tulips grow in gardens. "
                   "Gardens smell sweet.")
        mind.ask("What are tulips?")
        assert len(mind.last_thoughts) == 1              # default Low = 50 tokens

        mind.set_thinking_budget(100, ThinkingLevel.LOW)
        assert mind.thinking_tokens_for(ThinkingLevel.LOW) == 100
        mind.ask("What are tulips?")
        assert len(mind.last_thoughts) == 2              # 100/50

    def test_automatic_thinking_uses_the_chosen_levels_budget(self):
        # The resolver only picks the LEVEL; the level's budget follows.
        # Deep questions resolve to High..Max — with those levels budgeted
        # to 50 (one thought) a deep question thinks SHALLOWER than small
        # talk budgeted to 150 (three thoughts).
        mind = make_mind(
            automatic=True,
            budgets=[(150, ThinkingLevel.LOW), (50, ThinkingLevel.HIGH),
                     (50, ThinkingLevel.VERY_HIGH), (50, ThinkingLevel.MAX)])
        mind.teach("Tulips are flowers. Tulips grow in gardens. "
                   "Gardens smell sweet.")
        mind.ask("hi")
        small_talk = len(mind.last_thoughts)
        assert small_talk == 3            # auto chose Low → Low's 150-token budget

        mind.ask("Why do flowers turn toward the sun, and how does the whole "
                 "process actually work inside the plant over a full day?")
        deep = len(mind.last_thoughts)
        assert deep <= small_talk, \
            f"the chosen level's budget must govern: deep={deep}, small={small_talk}"

    def test_mind_chains(self):
        mind = make_mind()
        assert mind.set_thinking_budget(80, "high") is mind
