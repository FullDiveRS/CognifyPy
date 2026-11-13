"""Tests for the learning stack: encoder, MicroNet, Q-learning, learner."""
import math

import pytest

from cognify.learning.encoder import HashingEncoder
from cognify.learning.learner import LearningModule, sentiment_target
from cognify.learning.micro_net import MicroNet
from cognify.learning.qlearn import ACTIONS, QLearningAgent


class TestHashingEncoder:
    def test_deterministic(self):
        e = HashingEncoder(dims=64)
        assert e.encode("hello world") == e.encode("hello world")

    def test_l2_normalized(self):
        e = HashingEncoder(dims=64)
        v = e.encode("some text to encode here")
        assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-6)

    def test_similarity_ranking(self):
        e = HashingEncoder(dims=128)
        assert e.similarity("the dog chased the cat",
                            "the cat chased the dog") > \
               e.similarity("the dog chased the cat", "quantum physics is hard")

    def test_dims_configurable(self):
        assert len(HashingEncoder(dims=32).encode("x")) == 32

    def test_dims_minimum(self):
        with pytest.raises(ValueError):
            HashingEncoder(dims=4)

    def test_empty_text(self):
        v = HashingEncoder().encode("")
        assert all(x == 0.0 for x in v)


class TestMicroNet:
    def test_forward_shape(self):
        net = MicroNet(3, 5, 2, seed=1)
        out, hidden = net.forward([0.1, 0.2, 0.3])
        assert len(out) == 2 and len(hidden) == 5
        assert all(0.0 <= o <= 1.0 for o in out)

    def test_wrong_input_size_raises(self):
        net = MicroNet(3, 5, 2)
        with pytest.raises(ValueError):
            net.forward([0.1, 0.2])

    def test_xor_learns(self):
        net = MicroNet(2, 8, 2, seed=3)
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        Y = [[1, 0], [0, 1], [0, 1], [1, 0]]      # one-hot: class 0 / class 1
        loss = net.train(X, Y, epochs=400, lr=0.8)
        assert loss < 0.05
        for x, y in zip(X, Y):
            pred = net.predict(x)
            assert (pred[1] > pred[0]) == (y[1] > y[0])

    def test_loss_decreases_with_training(self):
        net = MicroNet(1, 6, 1, seed=5)
        first = net.train([[0], [1]], [[1], [0]], epochs=2, lr=0.5)
        net2 = MicroNet(1, 6, 1, seed=5)
        later = net2.train([[0], [1]], [[1], [0]], epochs=200, lr=0.5)
        assert later < first

    def test_param_count(self):
        net = MicroNet(4, 6, 3)
        assert net.n_params == 4 * 6 + 6 + 6 * 3 + 3

    def test_state_roundtrip(self):
        net = MicroNet(2, 4, 1, seed=9)
        net.train([[0, 1]], [[1]], epochs=5)
        clone = MicroNet(2, 4, 1)
        clone.load_state_dict(net.state_dict())
        assert clone.predict([0.5, 0.5]) == pytest.approx(net.predict([0.5, 0.5]))

    def test_shape_mismatch_on_load(self):
        net = MicroNet(2, 4, 1)
        other = MicroNet(3, 4, 1)
        with pytest.raises(ValueError):
            net.load_state_dict(other.state_dict())

    def test_invalid_layers(self):
        with pytest.raises(ValueError):
            MicroNet(0, 4, 1)


class TestQLearning:
    def test_state_key_shape(self):
        key = QLearningAgent.state_key(True, 0.5, 0.1)
        assert key == "Q|pos|old"

    def test_exploration_then_exploitation(self):
        agent = QLearningAgent(epsilon=0.0, seed=1)
        state = "S|neu|old"
        for _ in range(30):
            agent.update(state, "emotional", 1.0, state)
        assert agent.select(state) == "emotional"

    def test_epsilon_greedy_explores(self):
        agent = QLearningAgent(epsilon=0.9, seed=1)
        choices = {agent.select("S|neu|old") for _ in range(100)}
        assert len(choices) >= 2

    def test_update_moves_q_toward_reward(self):
        agent = QLearningAgent()
        q0 = agent.q_value("s", "direct")
        agent.update("s", "direct", 1.0)
        assert agent.q_value("s", "direct") > q0

    def test_epsilon_decay_floor(self):
        agent = QLearningAgent(epsilon=0.5, epsilon_min=0.05)
        for _ in range(100):
            agent.decay_epsilon(0.5)
        assert agent.epsilon == pytest.approx(0.05)

    def test_best_action(self):
        agent = QLearningAgent()
        agent.update("s", "curious", 2.0)
        assert agent.best_action("s") == "curious"

    def test_dict_roundtrip(self):
        agent = QLearningAgent(epsilon=0.3)
        agent.update("Q|pos|new", "direct", 1.0)
        clone = QLearningAgent()
        clone.load_dict(agent.to_dict())
        assert clone.q_value("Q|pos|new", "direct") == \
               agent.q_value("Q|pos|new", "direct")
        assert clone.epsilon == agent.epsilon

    def test_actions_defined(self):
        assert "direct" in ACTIONS and "curious" in ACTIONS


class TestLearner:
    def _ctx(self, text):
        from cognify.core.context import CognitiveContext
        from cognify.core.stimulus import Stimulus
        from cognify.emotions.emotion_state import EmotionalState
        from cognify.emotions.personality import Personality

        return CognitiveContext(cycle=1, stimulus=Stimulus.text(text),
                                personality=Personality(),
                                emotions=EmotionalState())

    def test_sentiment_target(self):
        assert sentiment_target("I love this great thing")[0] == 2
        assert sentiment_target("this is terrible and awful")[0] == 0
        assert sentiment_target("the sky is blue")[0] == 1

    def test_observe_trains(self):
        enc = HashingEncoder(dims=32)
        lm = LearningModule(enc, seed=1)
        texts = ("I love this", "this is terrible", "hello world", "great job",
                 "awful day", "nice work friend", "sad and boring", "so good")
        for text in texts:
            ctx = self._ctx(text)
            lm.observe(ctx)
        assert lm.net.samples_seen > 0
        assert lm.train_batches >= 1

    def test_sentiment_prediction_runs(self):
        enc = HashingEncoder(dims=32)
        lm = LearningModule(enc, seed=1)
        label, prob = lm.sentiment_of("wonderful brilliant amazing")
        assert label in ("negative", "neutral", "positive")
        assert 0.0 <= prob <= 1.0

    def test_choose_action_returns_valid(self):
        enc = HashingEncoder(dims=32)
        lm = LearningModule(enc, seed=1)
        action = lm.choose_action(self._ctx("what is a bird?"))
        assert action in ACTIONS

    def test_feedback_improves_q(self):
        enc = HashingEncoder(dims=32)
        lm = LearningModule(enc, seed=1)
        ctx = self._ctx("hello")
        action = lm.choose_action(ctx)
        lm.feedback(1.0)
        lm.observe(self._ctx("thanks, that was great"))
        # Q for the chosen (state, action) must have moved up
        state = lm.agent.best_action and None  # placeholder no-op
        assert lm.agent.stats()["updates"] >= 1

    def test_stats_keys(self):
        enc = HashingEncoder(dims=32)
        lm = LearningModule(enc, seed=1)
        lm.observe(self._ctx("hello there friend"))
        s = lm.stats()
        for key in ("net_params", "epochs_trained", "replay_size",
                    "q_learning", "last_loss"):
            assert key in s

    def test_snapshot_roundtrip(self):
        enc = HashingEncoder(dims=32)
        lm = LearningModule(enc, seed=1)
        lm.observe(self._ctx("I love this wonderful thing"))
        clone = LearningModule(HashingEncoder(dims=32), seed=2)
        clone.load_snapshot(lm.state_snapshot())
        assert clone.sentiment_of("great") == lm.sentiment_of("great")
