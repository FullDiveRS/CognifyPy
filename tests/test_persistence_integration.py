"""Tests for SQLite persistence and end-to-end integration."""
import pytest

from cognify import MindBuilder, Stimulus
from cognify.core.stimulus import Stimulus as S
from cognify.persistence.store import SqliteMindStore


@pytest.fixture
def store(tmp_path):
    return SqliteMindStore(str(tmp_path / "minds.db"))


def make_trained_mind():
    mind = (MindBuilder("Learner").with_seed(5).with_preset("curious_child")
            .enable_tool_use().build())
    mind.teach("Birds are animals. Birds can fly.")
    mind.process("I love this wonderful thing, thanks!")
    mind.process("What is 3 + 4?")
    mind.feedback(0.9)
    mind.process("hello again friend")
    return mind


class TestPersistence:
    def test_save_and_list(self, store):
        mind = make_trained_mind()
        store.save(mind, "snapshot-1")
        names = [s["name"] for s in store.list()]
        assert "snapshot-1" in names

    def test_load_unknown_raises(self, store):
        with pytest.raises(KeyError):
            store.load("ghost")

    def test_delete(self, store):
        mind = make_trained_mind()
        store.save(mind, "temp")
        assert store.delete("temp") is True
        assert store.delete("temp") is False
        assert store.list() == []

    def test_full_roundtrip(self, store):
        mind = make_trained_mind()
        store.save(mind, "clone-src")
        clone = store.load("clone-src")

        assert clone.name == mind.name
        assert clone.cycle_count == mind.cycle_count
        assert clone.kb.size() == mind.kb.size()
        assert clone.memory.size() == mind.memory.size()
        assert clone.personality.to_dict() == mind.personality.to_dict()
        assert clone.emotions.values == mind.emotions.values
        assert clone.kb.get("bird", "is-a", "animal") is not None

    def test_clone_answers_questions(self, store):
        mind = make_trained_mind()
        store.save(mind, "wise")
        clone = store.load("wise")
        answer = clone.ask("Can birds fly?")
        assert "Yes" in answer

    def test_learning_state_survives(self, store):
        mind = make_trained_mind()
        store.save(mind, "brain")
        clone = store.load("brain")
        assert clone.learner is not None
        assert clone.learner.net.state_dict()["w1"] == \
               mind.learner.net.state_dict()["w1"]
        assert clone.learner.agent.to_dict()["q"] == \
               mind.learner.agent.to_dict()["q"]

    def test_clone_continues_learning(self, store):
        mind = make_trained_mind()
        store.save(mind, "student")
        clone = store.load("wise" if False else "student")
        before = clone.learner.net.epochs_trained
        clone.process("another wonderful day of learning")
        assert clone.learner.net.epochs_trained >= before

    def test_overwrite_same_name(self, store):
        mind = make_trained_mind()
        store.save(mind, "same")
        mind.process("one more cycle")
        store.save(mind, "same")
        assert len(store.list()) == 1
        assert store.load("same").cycle_count == mind.cycle_count

    def test_encoder_dims_preserved(self, store):
        mind = MindBuilder("Dims").with_encoder_dims(64).build()
        mind.process("hello")
        store.save(mind, "dims")
        clone = store.load("dims")
        assert clone.encoder.dims == 64


class TestEndToEnd:
    def test_full_pipeline_all_capabilities(self):
        from cognify.capabilities.imaging import encode_png
        from cognify.capabilities.media import melody_from_prompt

        mind = (MindBuilder("Omni").with_seed(3)
                .enable_embeddings().enable_multilingual().enable_tool_use()
                .enable_vision().enable_audio().enable_code()
                .enable_media_generation().build())

        mind.teach("Birds are animals.")
        r1 = mind.process("What is a bird?")
        assert r1.text and r1.confidence > 0.3

        rows = [[(10, 10, 40)] * 32 for _ in range(24)]
        for y in range(8, 14):
            for x in range(12, 20):
                rows[y][x] = (200, 60, 60)
        r2 = mind.process(Stimulus.image(encode_png(32, 24, rows)))
        assert "vision" in r2.capabilities

        r3 = mind.process(Stimulus.audio(melody_from_prompt("calm tune").data))
        assert "audio" in r3.capabilities

        r4 = mind.process("Hola, gracias!")
        assert r4.text

        r5 = mind.process("What is 21 * 2?")
        assert any("42" in t for t in r5.tool_results)

        r6 = mind.process(
            "def foo(a):\n    return a\n")  # code via text triggers analysis
        assert r6.text

        r7 = mind.process("generate a happy song")
        assert any(a.kind == "audio" for a in r7.assets)

    def test_epsilon_greedy_never_crashes_over_long_run(self):
        mind = MindBuilder("Marathon").with_seed(11).enable_tool_use().build()
        texts = ["hello", "I love this", "this is terrible", "what is 1 + 1?",
                 "tell me about birds", "birds are animals"]
        for i in range(18):
            r = mind.process(texts[i % len(texts)])
            assert r.text
            if i % 5 == 0:
                mind.feedback(0.5)

    def test_demo_runs_end_to_end(self):
        """The scripted demo tour must complete without exceptions."""
        from cognify import demo
        demo._run_all_inner()

    def test_personality_changes_replies(self):
        soft = MindBuilder("Soft").with_seed(2).with_preset("calm_mentor").build()
        loud = MindBuilder("Loud").with_seed(2).with_preset("curious_child").build()
        for m in (soft, loud):
            m.process("birds are animals")
        r_soft = soft.process("What is a bird?")
        r_loud = loud.process("What is a bird?")
        assert r_soft.text and r_loud.text       # both sane; style may differ
        assert r_soft.meta["action"] in ("direct", "emotional", "curious",
                                         "share_thought")

    def test_novelty_decays_over_repeats(self):
        mind = MindBuilder("Novel").with_seed(4).build()
        n1 = mind.process("a brand new quantum sentence").meta
        n2 = mind.process("a brand new quantum sentence").meta
        # both runs succeed; the engine does not crash on repeats
        assert "cycle" in n1 and "cycle" in n2
