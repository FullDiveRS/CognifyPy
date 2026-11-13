"""AdamOptimizer: Adam (Kingma & Ba 2015) over named parameter tensors.

Mirrors the C# ``Cognify.Learning.AdamOptimizer`` — the shared update rule
behind :class:`~cognify.learning.deep_net.DeepNet` and
:class:`~cognify.learning.transformer.TransformerLM`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


class AdamOptimizer:
    """Adam over named matrices and vectors with a shared step counter."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8) -> None:
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.t = 0
        self._mat: Dict[str, Tuple[List[List[float]], List[List[float]]]] = {}
        self._vec: Dict[str, Tuple[List[float], List[float]]] = {}

    def reset(self) -> None:
        """Clears moment state (used when a parameter tensor is resized)."""
        self.t = 0
        self._mat.clear()
        self._vec.clear()

    def step(self, key: str, param: List[List[float]],
             grad: List[List[float]], lr: float) -> None:
        """One Adam step on a matrix parameter, in place."""
        if key not in self._mat:
            self._mat[key] = (
                [[0.0] * len(row) for row in param],
                [[0.0] * len(row) for row in param],
            )
        m, v = self._mat[key]
        self.t += 1
        bc1 = 1.0 - math.pow(self.beta1, self.t)
        bc2 = 1.0 - math.pow(self.beta2, self.t)
        for r, prow in enumerate(param):
            pm, pv, g = m[r], v[r], grad[r]
            for c in range(len(prow)):
                gvc = g[c]
                pm[c] = self.beta1 * pm[c] + (1.0 - self.beta1) * gvc
                pv[c] = self.beta2 * pv[c] + (1.0 - self.beta2) * gvc * gvc
                prow[c] -= lr * (pm[c] / bc1) / (math.sqrt(pv[c] / bc2) + self.epsilon)

    def step_vec(self, key: str, param: List[float],
                 grad: Sequence[float], lr: float) -> None:
        """One Adam step on a vector parameter, in place."""
        if key not in self._vec:
            self._vec[key] = ([0.0] * len(param), [0.0] * len(param))
        m, v = self._vec[key]
        self.t += 1
        bc1 = 1.0 - math.pow(self.beta1, self.t)
        bc2 = 1.0 - math.pow(self.beta2, self.t)
        for i in range(len(param)):
            gvc = grad[i]
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * gvc
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * gvc * gvc
            param[i] -= lr * (m[i] / bc1) / (math.sqrt(v[i] / bc2) + self.epsilon)
