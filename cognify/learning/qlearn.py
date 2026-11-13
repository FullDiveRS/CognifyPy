"""Q-learning agent: learns which response strategy pays off."""
from __future__ import annotations

import random
from typing import Dict, Tuple

from ..util import clamp

ACTIONS: Tuple[str, ...] = ("direct", "emotional", "curious", "share_thought",
                            "trained")


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    State is a compact key built from the situation (question? / valence /
    novelty); actions are response strategies. Rewards come from user
    feedback and inferred sentiment.
    """

    def __init__(self, epsilon: float = 0.25, alpha: float = 0.5,
                 gamma: float = 0.9, epsilon_min: float = 0.05,
                 seed: int = 7) -> None:
        self.epsilon = clamp(float(epsilon), 0.0, 1.0)
        self.epsilon_min = epsilon_min
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self._q: Dict[Tuple[str, str], float] = {}
        self._rng = random.Random(seed)
        self.updates = 0

    @staticmethod
    def state_key(has_question: bool, valence: float, novelty: float) -> str:
        v = "pos" if valence > 0.15 else "neg" if valence < -0.15 else "neu"
        n = "new" if novelty > 0.5 else "old"
        return f"{'Q' if has_question else 'S'}|{v}|{n}"

    def q_value(self, state: str, action: str) -> float:
        return self._q.get((state, action), 0.0)

    def select(self, state: str) -> str:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        best, best_q = ACTIONS[0], self._q.get((state, ACTIONS[0]), 0.0)
        for a in ACTIONS[1:]:
            q = self._q.get((state, a), 0.0)
            if q > best_q:
                best, best_q = a, q
        return best

    def update(self, state: str, action: str, reward: float,
               next_state: str = "") -> float:
        old = self._q.get((state, action), 0.0)
        future = 0.0
        if next_state:
            future = max(self._q.get((next_state, a), 0.0) for a in ACTIONS)
        target = reward + self.gamma * future
        self._q[(state, action)] = old + self.alpha * (target - old)
        self.updates += 1
        return self._q[(state, action)]

    def decay_epsilon(self, factor: float = 0.97) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * factor)

    def best_action(self, state: str) -> str:
        best, best_q = ACTIONS[0], self._q.get((state, ACTIONS[0]), 0.0)
        for a in ACTIONS[1:]:
            q = self._q.get((state, a), 0.0)
            if q > best_q:
                best, best_q = a, q
        return best

    def stats(self) -> dict:
        return {"states": len({s for s, _ in self._q}),
                "pairs": len(self._q), "epsilon": round(self.epsilon, 3),
                "updates": self.updates}

    def to_dict(self) -> dict:
        return {"q": {f"{s}|{a}": v for (s, a), v in self._q.items()},
                "epsilon": self.epsilon, "updates": self.updates}

    def load_dict(self, data: dict) -> None:
        self._q = {}
        for key, value in data.get("q", {}).items():
            # state keys contain '|', but actions never do: split once from the right
            s, a = key.rsplit("|", 1)
            self._q[(s, a)] = float(value)
        self.epsilon = clamp(float(data.get("epsilon", 0.25)), 0.0, 1.0)
        self.updates = int(data.get("updates", 0))
