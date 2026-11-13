"""Tests for the deep-learning core: AdamOptimizer + DeepNet."""
import pytest

from cognify.learning.adam import AdamOptimizer
from cognify.learning.deep_net import DeepNet, gelu, gelu_grad


class TestAdam:
    def test_steps_and_resets(self):
        adam = AdamOptimizer()
        p = [1.0, 1.0]
        g = [0.5, -0.5]
        adam.step_vec("p", p, g, 0.1)
        assert adam.t == 1
        assert p[0] != 1.0
        adam.reset()
        assert adam.t == 0

    def test_matrix_step(self):
        adam = AdamOptimizer()
        p = [[1.0, 1.0], [1.0, 1.0]]
        g = [[0.1, -0.1], [0.2, 0.0]]
        adam.step("w", p, g, 0.05)
        assert p[0][0] < 1.0


class TestGelu:
    def test_known_values(self):
        assert abs(gelu(0.0)) < 1e-9
        assert 0.841 <= gelu(1.0) <= 0.842
        assert -0.159 <= gelu(-1.0) <= -0.158
        assert gelu(10.0) > 9.9

    def test_grad_finite(self):
        for x in (-3.0, -1.0, 0.0, 0.5, 4.0):
            assert -1.0 < gelu_grad(x) < 2.0


class TestDeepNet:
    def test_learns_xor(self):
        net = DeepNet(2, [8, 8], 1, hidden_act="gelu", output_act="sigmoid",
                      dropout=0.0, seed=11)
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        Y = [[0], [1], [1], [0]]
        net.train(X, Y, epochs=500, lr=0.05, batch_size=4)
        assert net.predict([0, 0])[0] < 0.35
        assert net.predict([0, 1])[0] > 0.65
        assert net.predict([1, 0])[0] > 0.65
        assert net.predict([1, 1])[0] < 0.35

    def test_softmax_classification(self):
        net = DeepNet(6, [12], 3, hidden_act="relu", output_act="softmax", seed=5)
        X = [[1, 0, 0, 0, 0, 0], [0.9, 0.1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0], [0, 0.9, 0.1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0], [0, 0, 0.9, 0.1, 0, 0]]
        classes = [0, 0, 1, 1, 2, 2]
        loss = net.train_classes(X, classes, epochs=250, lr=0.03, batch_size=6)
        assert loss < 0.5
        for x, c in zip(X, classes):
            probs = net.predict(x)
            assert probs.index(max(probs)) == c

    def test_training_reduces_loss(self):
        net = DeepNet(4, [6, 6], 2, hidden_act="tanh", output_act="sigmoid", seed=3)
        X = [[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]]
        Y = [[1, 0], [0, 1], [1, 0], [0, 1]]
        first = net.train(X, Y, epochs=1, lr=0.05)
        last = net.train(X, Y, epochs=150, lr=0.05)
        assert last < first
        assert net.epochs_trained == 151

    def test_parameter_count_exact(self):
        net = DeepNet(4, [3], 2)
        assert net.n_params == 3 * 4 + 3 + 2 * 3 + 2
        assert net.depth == 2
        assert net.units == [3, 2]

    def test_predict_deterministic_with_dropout(self):
        net = DeepNet(4, [8, 8], 2, dropout=0.5, seed=9)
        x = [0.2, 0.8, 0.5, 0.1]
        assert net.predict(x) == net.predict(x)

    def test_serialization_roundtrip(self):
        net = DeepNet(5, [7, 4], 3, hidden_act="gelu", dropout=0.2, seed=21)
        net.train([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
                  [[1, 0, 0], [0, 0, 1]], epochs=30, lr=0.05)
        restored = DeepNet.from_dict(net.to_dict())
        x = [1, 2, 3, 4, 5]
        assert net.predict(x) == restored.predict(x)
        assert restored.n_params == net.n_params

    def test_rejects_invalid_shapes(self):
        with pytest.raises(ValueError):
            DeepNet(0, [4], 1)
        with pytest.raises(ValueError):
            DeepNet(3, [0], 1)
        net = DeepNet(3, [4], 2)
        with pytest.raises(ValueError):
            net.predict([1, 2, 3, 4])
        with pytest.raises(ValueError):
            DeepNet(3, [4], 2, hidden_act="quantum")
