"""REASONING TOKEN BUDGETS AND LEVELS, LINED UP WITH THINKING — the
reasoning phase now shares the thinking ladder: the same five levels, the
same ceiling, the same default budgets (50/100/200/300/400), a global
one-size budget override, a per-level ladder, and a level→depth ladder for
forward chaining (Medium keeps the historical 6 passes). Its explanation
trace appears ALONGSIDE thinking in the chain of thought, and the exported
GGUF carries thinking + reasoning levels (llama.cpp effort names) with both
budget ladders in the manifest. Mirrors the C# ReasoningBudgetTests."""
import json

import pytest

from cognify import MindBuilder
from cognify.core.options import (DEFAULT_LEVEL_BUDGETS, ReasoningCeiling,
                                  ReasoningLevel, ReasoningOptions,
                                  ThinkingCeiling, ThinkingLevel,
                                  ThinkingOptions, effort_name)


def make_mind(**reasoning_kwargs) -> object:
    builder = (MindBuilder("Reasoner")
               .with_reasoning(**reasoning_kwargs)
               .with_memory()
               .with_thinking())
    return builder.build()


class TestReasoningLadder:
    def test_default_budgets_match_thinking_level_for_level(self):
        for level in ThinkingLevel:
            assert ReasoningOptions().tokens_for(
                ReasoningLevel.from_thinking(level)) == \
                ThinkingOptions().tokens_for(level)
        r = ReasoningOptions()
        assert r.level is ReasoningLevel.MEDIUM
        assert r.ceiling is ReasoningCeiling.ONLY5
        assert r.tokens_for(ReasoningLevel.MEDIUM) == 100

    def test_reasoning_level_ladder_mirrors_thinking_rung_for_rung(self):
        # ReasoningLevel is its own enum with the SAME rungs in the SAME
        # order — the cross-ladder maps must round-trip 1:1.
        assert list(ReasoningLevel) == list(ThinkingLevel)
        for level in ThinkingLevel:
            mapped = ReasoningLevel.from_thinking(level)
            assert mapped.value == level.value
            assert mapped.to_thinking() is level
        assert ReasoningCeiling.from_thinking(ThinkingCeiling.ONLY4) \
            is ReasoningCeiling.ONLY4

    def test_global_budget_overrides_every_level(self):
        r = ReasoningOptions().budget(120)
        for level in ReasoningLevel:
            assert r.tokens_for(level) == 120
        # budget(0) clears the global — the ladder takes over again.
        assert r.budget(0).tokens_for(ReasoningLevel.MAX) == 400

    def test_per_level_budget_writes_and_clears_with_ceiling_clamp(self):
        r = ReasoningOptions().up_to(ReasoningCeiling.ONLY2)
        r.budget(600, ReasoningLevel.MAX)
        assert r.tokens_for(ReasoningLevel.MAX) == 600     # clamped onto Medium
        assert r.tokens_for(ReasoningLevel.MEDIUM) == 600
        r.budget(0, ReasoningLevel.MEDIUM)                 # zero clears the entry
        assert r.tokens_for(ReasoningLevel.MEDIUM) == 100  # default ladder again

    def test_level_sizes_depth_ladder_and_depth_pins_it(self):
        r = ReasoningOptions()
        assert r.effective_depth == 6                     # Medium keeps history
        r.at(ReasoningLevel.LOW)
        assert r.effective_depth == 2
        r.at(ReasoningLevel.MAX)
        assert r.max_inference_depth == 12
        r.depth(4)                                        # explicit pin wins…
        assert r.effective_depth == 4
        assert r.depth_is_custom is True
        r.at(ReasoningLevel.LOW)                          # …and stops the ladder
        assert r.effective_depth == 4

    def test_ceiling_clamps_the_level(self):
        r = ReasoningOptions().up_to(ReasoningCeiling.ONLY2).at(ReasoningLevel.MAX)
        assert r.level is ReasoningLevel.MEDIUM

    def test_effort_names_align_with_llama_cpp(self):
        ladder = [effort_name(level) for level in ReasoningLevel]
        assert ladder == ["low", "medium", "high", "xhigh", "max"]

    def test_round_trips_through_a_snapshot(self):
        r = (ReasoningOptions().at(ReasoningLevel.HIGH)
             .budget(150, ReasoningLevel.HIGH).depth(5))
        restored = ReasoningOptions.from_dict(json.loads(json.dumps(r.to_dict())))
        assert restored.level is ReasoningLevel.HIGH
        assert restored.tokens_for(ReasoningLevel.HIGH) == 150
        assert restored.effective_depth == 5
        assert restored.depth_is_custom is True


class TestReasoningCycle:
    def test_enable_flips_the_reasoning_switch(self):
        # The builder's enabled flag AND the options switch stay in sync —
        # the exporter reads the options switch for cognify.reasoning.enabled.
        r = ReasoningOptions()
        assert r.enabled is True
        r.enable(False)
        assert r.enabled is False
        mind = (MindBuilder("Switched")
                .with_reasoning(enabled=False).with_memory().with_thinking()
                .build())
        assert mind.reasoning_options.enabled is False
        assert mind.flags.get("reasoning") is False

    def test_reasoning_trace_appears_alongside_thinking_in_the_chain(self):
        mind = make_mind()
        mind.teach("birds are animals")
        response = mind.process("animals are living things")   # learns in-cycle

        # reasoning spent its budget and produced a trace line…
        assert any(t[1].startswith("Reasoning: ") for t in response.thoughts)
        assert any(t[0] == "hypothesize" and t[1].startswith("Reasoning: ")
                   for t in response.thoughts)
        # …and thinking's own stages still run alongside it.
        assert any(t[0] == "perceive" for t in response.thoughts)
        assert any("Learned" in t[1] for t in response.thoughts)
        assert response.learned_facts

    def test_tiny_budget_trims_the_trace_but_not_the_knowledge(self):
        mind = make_mind(budget=1)        # one token — nothing fits
        mind.teach("birds are animals")
        response = mind.process("animals are living things")
        # The trace is trimmed (no reasoning steps survive)…
        assert not any(t[1].startswith("Reasoning: ") for t in response.thoughts)
        # …but the knowledge base is untouched.
        assert response.learned_facts
        assert mind.reasoning_tokens_for(ReasoningLevel.MEDIUM) == 1

    def test_generous_budget_keeps_the_trace(self):
        mind = make_mind(budget=200)
        mind.teach("birds are animals")
        response = mind.process("animals are living things")
        assert any(t[1].startswith("Reasoning: Learned") for t in response.thoughts)

    def test_trace_survives_thinking_disabled(self):
        # No thinking flag: no chain of thought at all, but reasoning still
        # ran and learned — the trace rides on the reasoning step itself.
        mind = (MindBuilder("Reasoner")
                .with_reasoning()
                .with_memory()
                .with_thinking(False)
                .build())
        mind.teach("birds are animals")
        response = mind.process("animals are living things")
        assert response.learned_facts

    def test_runtime_budget_and_level_apis(self):
        mind = make_mind()
        mind.set_reasoning_budget(1)
        mind.teach("birds are animals")
        trimmed = mind.process("animals are living things")
        assert not any(t[1].startswith("Reasoning: ") for t in trimmed.thoughts)

        mind.set_reasoning_budget(0)             # ladder again
        mind.set_reasoning_level("high")         # depth 8, 200-token trace
        assert mind.reasoning_options.effective_depth == 8
        assert mind.reasoning_tokens_for(ThinkingLevel.HIGH) == 200
        kept = mind.process("fish are animals")
        assert any(t[1].startswith("Reasoning: ") for t in kept.thoughts)


class TestCognitiveLevel:
    def test_with_cognitive_level_sets_both_modules_when_enabled(self):
        mind = (MindBuilder("Leveled")
                .with_reasoning().with_memory().with_thinking()
                .with_cognitive_level("veryhigh")
                .build())
        assert mind.thinking_options.level is ThinkingLevel.VERY_HIGH
        assert mind.reasoning_options.level is ReasoningLevel.VERY_HIGH
        assert mind.reasoning_options.max_inference_depth == 10

    def test_with_cognitive_level_clamps_and_skips_disabled_modules(self):
        mind = (MindBuilder("Clamped")
                # Reasoning configured at High but built DISABLED — the
                # shared level call must leave its options untouched.
                .with_reasoning(level="high", enabled=False)
                .with_thinking()
                .with_cognitive_level("max", ceiling="only2")
                .build())
        assert mind.thinking_options.level is ThinkingLevel.MEDIUM       # clamped
        assert mind.reasoning_options.level is ReasoningLevel.HIGH       # skipped
        assert mind.reasoning_options.ceiling is ReasoningCeiling.ONLY5  # skipped


class TestExportLevels:
    def test_exported_gguf_carries_levels_and_budget_ladders_for_both_modules(
            self, tmp_path):
        mind = (MindBuilder("ExportLevels")
                .only("thinking", "reasoning", "memory")
                .build())
        corpus = tmp_path / "levels.txt"
        corpus.write_text(
            "ExportLevels is quiet.\nExportLevels likes code.\nThe sky is blue.\n",
            encoding="utf-8")
        mind.train_file(str(corpus))
        mind.train_deep_model(epochs=1)

        from cognify.export import MindExporter
        from cognify.export.exporter import export_gguf_llama_cpp
        path = tmp_path / "levels.gguf"
        export_gguf_llama_cpp(mind, str(path))

        from cognify.export.gguf import read_gguf
        gguf = read_gguf(str(path))

        # KV metadata — llama.cpp-aligned effort names + token budgets.
        assert gguf.metadata["cognify.reasoning.enabled"]
        assert gguf.metadata["cognify.reasoning.level"] == "medium"
        assert gguf.metadata["cognify.reasoning.budget"] == 100
        assert gguf.metadata["cognify.thinking.budget"] == 100

        # Sampling profile — tiny vocabulary → the tightest tier: low
        # temperature, strong repetition penalty, hard reply cap. Float32
        # storage rounds the values — compare approximately.
        assert gguf.metadata["cognify.sampling.temperature"] == pytest.approx(0.7)
        assert gguf.metadata["cognify.sampling.top_p"] == pytest.approx(0.9)
        assert gguf.metadata["cognify.sampling.top_k"] == 40
        assert gguf.metadata["cognify.sampling.repeat_penalty"] == pytest.approx(1.2)
        assert gguf.metadata["cognify.sampling.max_tokens"] == 128

        # Manifest — riding in the chat template, readable from /props:
        # level ladders + per-level budgets for BOTH modules.
        template = gguf.metadata["tokenizer.chat_template"]
        manifest = json.loads(
            template.split("cognify_manifest = '")[1].split("' %}")[0]
            .replace("\\u0027", "'"))
        thinking = manifest["thinking"]
        reasoning = manifest["reasoning"]
        assert reasoning["level"] == "medium"
        assert reasoning["budget"] == 100
        assert reasoning["levels"][0] == "low"
        assert reasoning["levels"][3] == "xhigh"      # VeryHigh → xhigh
        assert reasoning["levels"][4] == "max"
        assert reasoning["budgets"]["low"] == 50
        assert thinking["budgets"]["max"] == 400
        assert thinking["level"] == gguf.metadata["cognify.thinking.level"]
