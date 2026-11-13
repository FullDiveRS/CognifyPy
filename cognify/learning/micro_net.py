"""MicroNet: a small but real MLP with backpropagation and SGD."""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from ..util import sigmoid, tanh


class MicroNet:
    """2-layer feedforward network (tanh hidden, sigmoid output, MSE loss).

    Trained with genuine backpropagation and gradient descent — small enough
    to run anywhere, real enough to actually learn (e.g. XOR, sentiment).
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 42) -> None:
        if n_in < 1 or n_hidden < 1 or n_out < 1:
            raise ValueError("layer sizes must be >= 1")
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        rng = random.Random(seed)
        bound1 = (6.0 / (n_in + n_hidden)) ** 0.5
        bound2 = (6.0 / (n_hidden + n_out)) ** 0.5
        self.w1: List[List[float]] = [[rng.uniform(-bound1, bound1) for _ in range(n_in)]
                                      for _ in range(n_hidden)]
        self.b1: List[float] = [0.0] * n_hidden
        self.w2: List[List[float]] = [[rng.uniform(-bound2, bound2) for _ in range(n_hidden)]
                                      for _ in range(n_out)]
        self.b2: List[float] = [0.0] * n_out
        self.epochs_trained = 0
        self.samples_seen = 0

    # ------------------------------------------------------------------ forward
    def forward(self, x: Sequence[float]) -> Tuple[List[float], List[float]]:
        """Returns (output, hidden_activations)."""
        if len(x) != self.n_in:
            raise ValueError(f"expected input of size {self.n_in}, got {len(x)}")
        h = []
        for j in range(self.n_hidden):
            z = self.b1[j]
            row = self.w1[j]
            for i in range(self.n_in):
                z += row[i] * x[i]
            h.append(tanh(z))
        o = []
        for k in range(self.n_out):
            z = self.b2[k]
            row = self.w2[k]
            for j in range(self.n_hidden):
                z += row[j] * h[j]
            o.append(sigmoid(z))
        return o, h

    def predict(self, x: Sequence[float]) -> List[float]:
        out, _ = self.forward(x)
        return out

    # ------------------------------------------------------------------ training
    def train_step(self, x: Sequence[float], target: Sequence[float],
                   lr: float = 0.3) -> float:
        """One backprop step on a single sample; returns squared error."""
        if len(target) != self.n_out:
            raise ValueError(f"expected target of size {self.n_out}, got {len(target)}")
        o, h = self.forward(x)

        # output deltas: (o - t) * sigmoid'
        d_out = [(o[k] - target[k]) * o[k] * (1.0 - o[k]) for k in range(self.n_out)]
        # hidden deltas: sum(w2 * d_out) * tanh'
        d_hid = []
        for j in range(self.n_hidden):
            s = sum(self.w2[k][j] * d_out[k] for k in range(self.n_out))
            d_hid.append(s * (1.0 - h[j] * h[j]))

        # gradient descent updates
        for k in range(self.n_out):
            for j in range(self.n_hidden):
                self.w2[k][j] -= lr * d_out[k] * h[j]
            self.b2[k] -= lr * d_out[k]
        for j in range(self.n_hidden):
            for i in range(self.n_in):
                self.w1[j][i] -= lr * d_hid[j] * x[i]
            self.b1[j] -= lr * d_hid[j]

        self.samples_seen += 1
        return sum((o[k] - target[k]) ** 2 for k in range(self.n_out)) / self.n_out

    def train(self, X: List[Sequence[float]], Y: List[Sequence[float]],
              epochs: int = 50, lr: float = 0.3, shuffle: bool = True) -> float:
        """Train for N epochs; returns final mean loss over the last epoch."""
        rng = random.Random(0)
        indices = list(range(len(X)))
        loss = 0.0
        for _ in range(max(0, epochs)):
            if shuffle:
                rng.shuffle(indices)
            loss = 0.0
            for idx in indices:
                loss += self.train_step(X[idx], Y[idx], lr)
            loss /= max(1, len(X))
            self.epochs_trained += 1
        return loss

    # ------------------------------------------------------------------ state
    @property
    def n_params(self) -> int:
        return (self.n_hidden * self.n_in + self.n_hidden +
                self.n_out * self.n_hidden + self.n_out)

    def state_dict(self) -> dict:
        return {"n_in": self.n_in, "n_hidden": self.n_hidden, "n_out": self.n_out,
                "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2,
                "epochs_trained": self.epochs_trained, "samples_seen": self.samples_seen}

    def load_state_dict(self, state: dict) -> None:
        if (state["n_in"], state["n_hidden"], state["n_out"]) != (
                self.n_in, self.n_hidden, self.n_out):
            raise ValueError("network shape mismatch")
        self.w1 = [list(r) for r in state["w1"]]
        self.b1 = list(state["b1"])
        self.w2 = [list(r) for r in state["w2"]]
        self.b2 = list(state["b2"])
        self.epochs_trained = state.get("epochs_trained", 0)
        self.samples_seen = state.get("samples_seen", 0)
