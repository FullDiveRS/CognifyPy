"""DeepNet: a genuinely DEEP feed-forward network — backprop + Adam.

Mirrors the C# ``Cognify.Learning.DeepNet`` exactly (same activations,
He initialization, inverted dropout, L2, mini-batch Adam, softmax/sigmoid/
linear heads). Where :class:`~cognify.learning.micro_net.MicroNet` is one
hidden layer deep, a DeepNet stacks an arbitrary number of hidden layers —
the classic "deep" in deep learning, small enough to train inside a
cognitive cycle in pure Python.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .adam import AdamOptimizer


def gelu(x: float) -> float:
    """GELU activation (tanh approximation), the transformer-era default."""
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(inner))


def gelu_grad(x: float) -> float:
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    t = math.tanh(inner)
    dt = (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * dt


_HIDDEN_ACTS = ("relu", "gelu", "tanh")
_OUTPUT_ACTS = ("sigmoid", "softmax", "linear")


class DeepNet:
    """Deep feed-forward network: input -> hidden layers -> output head.

    Targets: one-hot rows for a softmax head (cross-entropy loss), vectors
    in (0, 1) for sigmoid (MSE), or real vectors for linear (MSE).
    """

    def __init__(self, n_in: int, hidden: Sequence[int], n_out: int,
                 hidden_act: str = "gelu", output_act: str = "sigmoid",
                 dropout: float = 0.0, l2: float = 0.0, seed: int = 42) -> None:
        if n_in < 1 or n_out < 1:
            raise ValueError("input and output must be >= 1")
        if any(h < 1 for h in hidden):
            raise ValueError("hidden layer widths must be >= 1")
        hidden_act = hidden_act.lower()
        output_act = output_act.lower()
        if hidden_act not in _HIDDEN_ACTS or output_act not in _OUTPUT_ACTS:
            raise ValueError(f"unknown activation (hidden={hidden_act}, output={output_act})")

        self.n_in, self.n_out = n_in, n_out
        self.hidden_act, self.output_act = hidden_act, output_act
        self.dropout = min(0.9, max(0.0, dropout))
        self.l2 = max(0.0, l2)
        self._rng = random.Random(seed)
        self._adam = AdamOptimizer()

        self.weights: List[List[List[float]]] = []  # per layer: [units][fan_in]
        self.biases: List[List[float]] = []
        fan_in = n_in
        units = list(hidden) + [n_out]
        for li, width in enumerate(units):
            is_output = li == len(units) - 1
            std = 0.01 if (is_output and output_act == "softmax") else self._he_std(fan_in)
            layer = [[self._gauss(std) for _ in range(fan_in)] for _ in range(width)]
            self.weights.append(layer)
            self.biases.append([0.0] * width)
            fan_in = width

        self.epochs_trained = 0
        self.samples_seen = 0

    # ------------------------------------------------------------------ init
    @staticmethod
    def _he_std(fan_in: int) -> float:
        """He initialization (He et al. 2015) — stable ReLU/GELU depth."""
        return math.sqrt(2.0 / max(1, fan_in))

    def _gauss(self, std: float) -> float:
        u1 = 1.0 - self._rng.random()
        u2 = 1.0 - self._rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) * std

    # ------------------------------------------------------------------ shape
    @property
    def depth(self) -> int:
        return len(self.weights)

    @property
    def units(self) -> List[int]:
        return [len(w) for w in self.weights]

    @property
    def n_params(self) -> int:
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += sum(len(row) for row in w) + len(b)
        return total

    # ------------------------------------------------------------------ forward
    def _activate(self, act: str, x: float) -> float:
        if act == "relu":
            return x if x > 0 else 0.0
        if act == "gelu":
            return gelu(x)
        return math.tanh(x)

    def _activate_grad(self, act: str, z: float, a: float) -> float:
        if act == "relu":
            return 1.0 if z > 0 else 0.0
        if act == "gelu":
            return gelu_grad(z)
        return 1.0 - a * a

    def forward(self, x: Sequence[float],
                training: bool = False) -> Tuple[List[float], List[List[float]], List[List[float]], List[Optional[List[float]]]]:
        """Returns (output, activations, pre_activations, dropout_masks)."""
        if len(x) != self.n_in:
            raise ValueError(f"expected input of size {self.n_in}, got {len(x)}")
        activations: List[List[float]] = []
        pre_acts: List[List[float]] = []
        masks: List[Optional[List[float]]] = []
        a = list(x)
        depth = len(self.weights)
        for li in range(depth):
            is_output = li == depth - 1
            w, b = self.weights[li], self.biases[li]
            z: List[float] = []
            for u in range(len(w)):
                s = b[u]
                row = w[u]
                for i, av in enumerate(a):
                    s += row[i] * av
                z.append(s)
            if is_output:
                if self.output_act == "sigmoid":
                    out = [1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, zv)))) for zv in z]
                elif self.output_act == "softmax":
                    m = max(z) if z else 0.0
                    exps = [math.exp(zv - m) for zv in z]
                    total = sum(exps) or 1.0
                    out = [e / total for e in exps]
                else:
                    out = list(z)
            else:
                out = [self._activate(self.hidden_act, zv) for zv in z]
                if training and self.dropout > 0:
                    keep = 1.0 - self.dropout
                    mask = [0.0 if self._rng.random() < self.dropout else 1.0 / keep for _ in out]
                    out = [o * mk for o, mk in zip(out, mask)]
                    masks.append(mask)
                else:
                    masks.append(None)
            activations.append(out)
            pre_acts.append(z)
            a = out
        return a, activations, pre_acts, masks

    def predict(self, x: Sequence[float]) -> List[float]:
        out, _, _, _ = self.forward(x, training=False)
        return out

    def predict_batch(self, xs: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self.predict(x) for x in xs]

    # ------------------------------------------------------------------ training
    def train(self, X: List[Sequence[float]], Y: List[Sequence[float]],
              epochs: int = 50, lr: float = 0.05, batch_size: int = 32) -> float:
        """Mini-batch Adam training; returns the mean loss of the last epoch."""
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        if not X:
            return 0.0
        order = list(range(len(X)))
        loss = 0.0
        for _ in range(max(1, epochs)):
            self._rng.shuffle(order)
            loss = 0.0
            seen = 0
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                loss += self._train_batch([X[i] for i in idx], [Y[i] for i in idx], lr)
                seen += len(idx)
            loss /= max(1, seen)
            self.epochs_trained += 1
            self.samples_seen += len(X)
        return loss

    def train_classes(self, X: List[Sequence[float]], classes: List[int],
                      epochs: int = 50, lr: float = 0.05, batch_size: int = 32) -> float:
        """Convenience: class-index targets for a softmax head."""
        if self.output_act != "softmax":
            raise ValueError("class-index targets require a softmax head")
        Y = []
        for c in classes:
            if not 0 <= c < self.n_out:
                raise ValueError(f"class index {c} out of range")
            onehot = [0.0] * self.n_out
            onehot[c] = 1.0
            Y.append(onehot)
        return self.train(X, Y, epochs, lr, batch_size)

    def _train_batch(self, xs: List[List[float]], ys: List[List[float]], lr: float) -> float:
        depth = len(self.weights)
        grad_w = [[[0.0] * len(row) for row in layer] for layer in self.weights]
        grad_b = [[0.0] * len(b) for b in self.biases]
        batch_loss = 0.0

        for x, target in zip(xs, ys):
            out, acts, pres, masks = self.forward(x, training=True)
            n = 1.0 / self.n_out
            delta = [0.0] * self.n_out
            if self.output_act == "softmax":
                for o in range(self.n_out):
                    delta[o] = out[o] - target[o]
                    if target[o] > 0:
                        batch_loss -= math.log(max(out[o], 1e-12))
            else:
                for o in range(self.n_out):
                    err = out[o] - target[o]
                    batch_loss += err * err * n
                    d_act = out[o] * (1.0 - out[o]) if self.output_act == "sigmoid" else 1.0
                    delta[o] = err * d_act * n

            grad = delta
            for li in range(depth - 1, -1, -1):
                lower = x if li == 0 else acts[li - 1]
                for u in range(len(self.weights[li])):
                    d = grad[u]
                    gw = grad_w[li][u]
                    for i, lv in enumerate(lower):
                        gw[i] += d * lv
                    grad_b[li][u] += d
                if li > 0:
                    nxt = [0.0] * len(self.weights[li - 1])
                    for u in range(len(self.weights[li])):
                        d = grad[u]
                        row = self.weights[li][u]
                        for i, rv in enumerate(row):
                            nxt[i] += d * rv
                    mask = masks[li - 1]
                    for i in range(len(nxt)):
                        if mask is not None:
                            nxt[i] *= mask[i]
                        nxt[i] *= self._activate_grad(self.hidden_act, pres[li - 1][i], acts[li - 1][i])
                    grad = nxt

        batch = len(xs)
        for li in range(depth):
            for u in range(len(grad_w[li])):
                gw = grad_w[li][u]
                for i in range(len(gw)):
                    gw[i] /= batch
                    if self.l2 > 0:
                        gw[i] += self.l2 * self.weights[li][u][i]
            for u in range(len(grad_b[li])):
                grad_b[li][u] /= batch
            self._adam.step(f"w{li}", self.weights[li], grad_w[li], lr)
            self._adam.step_vec(f"b{li}", self.biases[li], grad_b[li], lr)

        return batch_loss

    # ------------------------------------------------------------------ state
    def to_dict(self) -> dict:
        return {
            "n_in": self.n_in, "n_out": self.n_out,
            "hidden": self.units[:-1],
            "hidden_act": self.hidden_act, "output_act": self.output_act,
            "dropout": self.dropout, "l2": self.l2,
            "weights": self.weights, "biases": self.biases,
            "epochs_trained": self.epochs_trained, "samples_seen": self.samples_seen,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "DeepNet":
        net = cls(int(state["n_in"]), list(state.get("hidden", [])), int(state["n_out"]),
                  state.get("hidden_act", "gelu"), state.get("output_act", "sigmoid"),
                  float(state.get("dropout", 0.0)), float(state.get("l2", 0.0)),
                  int(state.get("seed", 42)))
        if net.units != [len(w) for w in state["weights"]]:
            raise ValueError("network shape mismatch")
        net.weights = [[list(row) for row in layer] for layer in state["weights"]]
        net.biases = [list(b) for b in state["biases"]]
        net.epochs_trained = int(state.get("epochs_trained", 0))
        net.samples_seen = int(state.get("samples_seen", 0))
        return net
