"""Tests for the cognitive core: builder, mind loop, events, stimulus."""
import pytest

from cognify import EventBus, MindBuilder, Modality, Stimulus, ThoughtStage


def make_mind(**kw):
    return MindBuilder("TestMind").with_seed(7).build()


class TestBuilder:
    def test_builds_with_defaults(self):
        mind = make_mind()
        assert mind.name == "TestMind"
        assert mind.cycle_count == 0

    def test_fluent_chain(self):
        mind = (MindBuilder("X").with_seed(1).with_curiosity(0.9)
                .with_memory_capacity(50).with_encoder_dims(64).build())
        assert mind.name == "X"
        assert mind.memory.capacity == 50
        assert mind.encoder.dims == 64
        assert mind.personality.openness == pytest.approx(0.9)

    def test_preset(self):
        mind = MindBuilder().with_preset("calm_mentor").build()
        assert mind.personality.neuroticism < 0.5
        assert mind.personality.agreeableness > 0.7

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            MindBuilder().with_preset("nope")

    def test_random_personality_seeded(self):
        a = MindBuilder().with_random_personality(3).build().personality
        b = MindBuilder().with_random_personality(3).build().personality
        assert a.to_dict() == b.to_dict()

    def test_disable_module(self):
        mind = MindBuilder().with_emotions(False).build()
        r = mind.process("I love this!")
        assert "emotions" not in r.meta or r.emotions == {}

    def test_curiosity_clamped(self):
        mind = MindBuilder().with_curiosity(5.0).build()
        assert mind.personality.openness == 1.0


class TestStimulus:
    def test_text_factory(self):
        s = Stimulus.text("hi", salience=0.9)
        assert s.modality == Modality.TEXT and s.salience == 0.9

    def test_image_requires_data(self):
        with pytest.raises(ValueError):
            Stimulus(modality=Modality.IMAGE)

    def test_data_must_be_bytes(self):
        with pytest.raises(TypeError):
            Stimulus(data="not bytes")

    def test_is_empty(self):
        assert Stimulus.text("").is_empty
        assert not Stimulus.text("hi").is_empty


class TestMindLoop:
    def test_process_returns_response(self):
        mind = make_mind()
        r = mind.process("hello there")
        assert r.text
        assert 0.0 <= r.confidence <= 1.0
        assert mind.cycle_count == 1

    def test_seven_thought_stages_in_order(self):
        mind = make_mind()
        r = mind.process("what is life?")
        stages = [s for s, _, _ in r.thoughts]
        deliberate = [st.value for st in ThoughtStage
                      if st is not ThoughtStage.GENERATE]
        # the deliberate pass keeps its 7-stage order...
        assert stages[:len(deliberate)] == deliberate
        # ...and the thought generator's pre-speech thoughts come after it
        # (sized by the thinking level — the reply is the result that
        # arrives once they finish)
        assert stages[len(deliberate):] and \
            set(stages[len(deliberate):]) == {"generate"}

    def test_accepts_plain_string(self):
        mind = make_mind()
        assert mind.process("just a string").text

    def test_empty_stimulus_handled(self):
        mind = make_mind()
        r = mind.process("")
        assert r.text  # graceful fallback

    def test_events_fired(self):
        mind = make_mind()
        seen = []
        mind.events.subscribe("stimulus_received", lambda e: seen.append(e.type))
        mind.events.subscribe("response_composed", lambda e: seen.append(e.type))
        mind.process("hello")
        assert "stimulus_received" in seen and "response_composed" in seen

    def test_wildcard_subscription(self):
        mind = make_mind()
        count = []
        mind.events.subscribe("*", lambda e: count.append(1))
        mind.process("hello world")
        assert len(count) >= 3

    def test_event_history(self):
        mind = make_mind()
        mind.process("one")
        mind.process("two")
        types = [e.type for e in mind.events.history]
        assert types.count("stimulus_received") == 2

    def test_teach_returns_facts(self):
        mind = make_mind()
        facts = mind.teach("Birds are animals. Birds can fly.")
        assert len(facts) == 2
        assert mind.kb.size() == 2

    def test_teach_does_not_parse_questions(self):
        mind = make_mind()
        assert mind.teach("What is a bird?") == []

    def test_ask_answers_taught_question(self):
        mind = make_mind()
        mind.teach("Birds are animals.")
        assert "animal" in mind.ask("What is a bird?")

    def test_feedback_no_crash(self):
        mind = make_mind()
        mind.process("hi")
        mind.feedback(0.8)
        mind.process("hi again")     # reward consumed here

    def test_stats_shape(self):
        mind = make_mind()
        mind.teach("Cats are animals.")
        s = mind.stats()
        for key in ("cycles", "facts", "memories", "goals_open", "personality"):
            assert key in s
        assert s["facts"] == 1

    def test_response_emotions_snapshot(self):
        mind = make_mind()
        r = mind.process("I love this wonderful thing!")
        assert "joy" in r.emotions
        assert r.emotions["joy"] > 0


class TestCapabilitiesWiring:
    def test_embeddings_on_by_default(self):
        mind = make_mind()
        assert "embeddings" in [c.name for c in mind.capabilities]

    def test_disabled_capability_not_run(self):
        mind = MindBuilder().enable_embeddings(False).build()
        assert "embeddings" not in [c.name for c in mind.capabilities]

    def test_tool_use_registered(self):
        mind = MindBuilder().with_seed(1).enable_tool_use().build()
        assert "calculator" in mind.tools.names()
        r = mind.process("What is 12 * 12?")
        assert any("144" in t for t in r.tool_results)

    def test_capability_error_is_contained(self):
        from cognify.capabilities.base import Capability, Percept

        class Boom(Capability):
            name = "boom"
            description = "always fails"
            phase = "perceive"

            def can_handle(self, stimulus, ctx=None):
                return True

            def handle(self, stimulus, ctx=None):
                raise RuntimeError("boom")

        mind = MindBuilder().with_capability(Boom()).build()
        r = mind.process("hello")          # must not raise
        assert r.text
        errs = [e for e in mind.events.history if e.type == "capability_error"]
        assert errs and "boom" in errs[0].payload["error"]


class TestThoughtStreamUnit:
    def test_summary_and_conclusion(self):
        from cognify.core.thought_stream import ThoughtStream

        ts = ThoughtStream()
        ts.add(ThoughtStage.PERCEIVE, "a", 0.5)
        ts.add(ThoughtStage.CONCLUDE, "b", 0.9)
        assert ts.summary() == "perceive -> conclude"
        assert ts.conclusion.content == "b"

    def test_confidence_clamped(self):
        from cognify.core.thought_stream import Thought

        t = Thought(ThoughtStage.PERCEIVE, "x", 5.0)
        assert t.confidence == 1.0
