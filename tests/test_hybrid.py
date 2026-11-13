"""Tests for the hybrid decision engine: the style-conditioned neural
network learning (context, strategy, mood) -> reward, the adaptive
symbolic/neural blend, and persistence — mirrors the C# HybridEngineTests."""
import pytest

from cognify import MindBuilder
from cognify.core.gender import GenderOptions
from cognify.learning.encoder import HashingEncoder
from cognify.learning.learner import LearningModule, ACTION_PRIORS
from cognify.learning.micro_net import MicroNet
from cognify.learning.qlearn import ACTIONS


def _features(dim, fill=0.5):
    return [fill * ((i % 3) + 1) / 3.0 for i in range(dim)]


# ------------------------------------------------------------------ style net
def test_style_input_has_one_hot_and_mood():
    learner = LearningModule(HashingEncoder(dims=16), seed=1)
    inp = learner.style_input(_features(16), "curious", valence=-0.5, arousal=0.9)
    assert len(inp) == 16 + len(ACTIONS) + 2
    onehot = inp[16:16 + len(ACTIONS)]
    assert onehot[ACTIONS.index("curious")] == 1.0
    assert sum(onehot) == 1.0
    assert inp[-2] == -0.5
    assert inp[-1] == 0.9


def test_style_net_learns_preferred_action():
    # "curious" keeps paying off, "trained" keeps disappointing: the network
    # must rank curious above trained for the SAME features and mood.
    learner = LearningModule(HashingEncoder(dims=16), seed=7)
    feats = _features(16)
    for _ in range(60):
        inp_c = learner.style_input(feats, "curious", 0.2, 0.6)
        learner.style_net.train_step(inp_c, [0.95], lr=0.25)
        inp_t = learner.style_input(feats, "trained", 0.2, 0.6)
        learner.style_net.train_step(inp_t, [0.05], lr=0.25)
    pred_curious = learner.style_net.predict(learner.style_input(feats, "curious", 0.2, 0.6))[0]
    pred_trained = learner.style_net.predict(learner.style_input(feats, "trained", 0.2, 0.6))[0]
    assert pred_curious > pred_trained + 0.2


def test_hybrid_score_ranks_reinforced_action_higher():
    mind = MindBuilder("Hybrid").build()
    learner = mind.learner
    feats = learner.encoder.encode("let us talk about training")

    # Reinforce one action repeatedly with reward, another with punishment.
    for _ in range(40):
        learner._pending = ("S|neu|old", "curious", feats, 0.1, 0.5)
        learner._explicit_reward = 1.0
        learner.observe(type("C", (), {"text": "ok", "scratch": {}, "has_question": False,
                                       "emotions": type("E", (), {"valence": 0.1})(),
                                       "cycle": 1})())
    for _ in range(40):
        learner._pending = ("S|neu|old", "emotional", feats, 0.1, 0.5)
        learner._explicit_reward = -1.0
        learner.observe(type("C", (), {"text": "ok", "scratch": {}, "has_question": False,
                                       "emotions": type("E", (), {"valence": 0.1})(),
                                       "cycle": 1})())

    good = learner.hybrid_score(feats, "curious", "S|neu|old", 0.1, 0.5)
    bad = learner.hybrid_score(feats, "emotional", "S|neu|old", 0.1, 0.5)
    assert good > bad


def test_neural_weight_stays_in_bounds_and_adapts():
    learner = LearningModule(HashingEncoder(dims=16), seed=3)
    assert 0.25 <= learner.neural_weight <= 0.80
    # Perfect predictions push the weight toward the neural side.
    for _ in range(40):
        learner._adapt_neural_weight(0.9, 0.9)
    assert 0.25 <= learner.neural_weight <= 0.80
    assert learner.neural_weight > 0.45      # accurate -> more neural trust
    # Terrible predictions pull it back toward symbolic.
    for _ in range(60):
        learner._adapt_neural_weight(0.0, 1.0)
    assert 0.25 <= learner.neural_weight <= 0.80


# ------------------------------------------------------------------ mind-level
def test_mind_stats_report_hybrid_engine():
    mind = MindBuilder("Hybrid").build()
    mind.process("hello there, how are you doing today?")
    stats = mind.stats()["learning"]
    assert "hybrid" in stats
    assert 0.25 <= stats["hybrid"]["neural_weight"] <= 0.80
    assert stats["vocabulary"] >= 0


def test_hybrid_engine_state_survives_snapshot():
    import tempfile, os
    from cognify.persistence.store import SqliteMindStore
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteMindStore(os.path.join(tmp, "minds.db"))
        mind = MindBuilder("Hybrid").build()
        for _ in range(6):
            mind.process("wonderful things are happening today")
            mind.feedback(0.8)
        weight_before = mind.learner.neural_weight
        store.save(mind, "hy")
        clone = store.load("hy")
        assert clone.learner.style_net.state_dict()["w1"] == \
               mind.learner.style_net.state_dict()["w1"]
        assert abs(clone.learner.neural_weight - weight_before) < 1e-9
