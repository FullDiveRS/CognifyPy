"""LearningModule: wires the nets, encoder, and Q-agent into the mind loop.

The decision engine is a HYBRID:

- symbolic side: tabular Q-learning per (situation, strategy) plus the main
  net's context reward prediction;
- neural side: a style-conditioned MicroNet that learns how rewarding each
  response STRATEGY is in the current context and mood
  (text features + strategy one-hot + valence/arousal -> reward);
- adaptive blend: the better the neural predictions get, the more the mind
  trusts the neural side (``neural_weight`` drifts between 0.25 and 0.80).
"""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from ..core.context import CognitiveContext
from ..emotions.appraisal import PLEASANT, UNPLEASANT
from ..util import clamp, tokenize
from .encoder import HashingEncoder
from .lexicon import Lexicon
from .micro_net import MicroNet
from .qlearn import QLearningAgent, ACTIONS

SENTIMENTS = ("negative", "neutral", "positive")

# Symbolic priors per strategy (personality-free baseline fit), used as the
# prior term of the hybrid score. Mirrors the C# composer priors.
ACTION_PRIORS: Dict[str, float] = {
    "direct": 0.55,
    "emotional": 0.50,
    "curious": 0.60,
    "share_thought": 0.45,
    "trained": 0.50,
}


def sentiment_target(text: str) -> Tuple[int, float]:
    """Crude lexicon sentiment used as a training target.

    Returns (class_index, strength) where class_index indexes SENTIMENTS.
    """
    tokens = set(tokenize(text))
    pos = sum(weight for w, weight in PLEASANT.items() if " " not in w and w in tokens)
    neg = sum(weight for w, weight in UNPLEASANT.items() if " " not in w and w in tokens)
    if pos > neg:
        return 2, min(1.0, 0.5 + 0.2 * (pos - neg))
    if neg > pos:
        return 0, min(1.0, 0.5 + 0.2 * (neg - pos))
    return 1, 0.5


class LearningModule:
    """Self-learning coordinator (hybrid symbolic + neural).

    - Trains a MicroNet to predict the emotional tone of incoming text
      (real gradient descent on a replay buffer).
    - Trains a style-conditioned MicroNet to predict how rewarding each
      response strategy is in the current context and mood.
    - Blends neural predictions with Q-table evidence adaptively
      (``neural_weight``) — a real hybrid engine.
    - Picks up new words from the user in a :class:`Lexicon` so the mind
      literally adds words as it talks.
    - Applies reward on the *next* cycle, since user reactions arrive later.
    """

    def __init__(self, encoder: HashingEncoder, seed: int = 42,
                 replay_size: int = 256) -> None:
        self.encoder = encoder
        self.net = MicroNet(encoder.dims, max(8, encoder.dims // 2), 3, seed=seed)
        # Style-conditioned net: text features + strategy one-hot + mood.
        self.style_net = MicroNet(encoder.dims + len(ACTIONS) + 2, 12, 1, seed=seed)
        self.agent = QLearningAgent(seed=seed)
        self.replay: Deque[Tuple[List[float], List[float]]] = deque(maxlen=replay_size)
        # (state, action, features, valence, arousal) awaiting reward
        self._pending: Optional[Tuple[str, str, List[float], float, float]] = None
        self._explicit_reward: Optional[float] = None
        self.train_batches = 0
        self.last_loss = 0.0
        self.last_style_loss = 0.0
        self._last_scores: Dict[str, float] = {}
        self.rng = random.Random(seed)
        # Adaptive symbolic/neural blend (see AdaptNeuralWeight mirror).
        self.neural_weight = 0.45
        self._style_pred_err = 0.25
        # The mind's growing word vocabulary.
        self.lexicon = Lexicon()

    # ------------------------------------------------------------------ cycle API
    def observe(self, ctx: CognitiveContext) -> None:
        """Called near the end of each cycle: train + schedule Q updates."""
        # 1. sentiment training example from this stimulus
        cls, strength = sentiment_target(ctx.text)
        target = [0.0, 0.0, 0.0]
        target[cls] = strength
        x = self.encoder.encode(ctx.text) if ctx.text else [0.0] * self.encoder.dims
        self.replay.append((x, target))
        if len(self.replay) >= 4:
            self._replay_train()

        # 2. close out the previous cycle's Q update using new evidence
        if self._pending is not None:
            state, action, features, decision_valence, decision_arousal = self._pending
            reward = self._explicit_reward if self._explicit_reward is not None else 0.0
            reward += self._inferred_reward(ctx.text)
            new_state = self.state_for(ctx)
            self.agent.update(state, action, reward, new_state)
            self.agent.decay_epsilon()
            # Train the style-conditioned half of the hybrid engine on the
            # same outcome, and let its prediction accuracy steer how much
            # the mind trusts the neural side next time.
            target01 = clamp((reward + 1) / 2.0)
            inp = self.style_input(features, action, decision_valence, decision_arousal)
            predicted = self.style_net.predict(inp)[0]
            self.last_style_loss = self.style_net.train_step(inp, [target01], lr=0.25)
            self._adapt_neural_weight(predicted, target01)
            self._pending = None
        self._explicit_reward = None

    def _adapt_neural_weight(self, predicted: float, actual: float) -> None:
        """Accurate predictions push the blend toward the neural side (max
        0.80), poor ones back toward symbolic (min 0.25)."""
        err = abs(predicted - actual)
        self._style_pred_err += 0.1 * (err - self._style_pred_err)
        target = 0.75 if self._style_pred_err < 0.15 else \
                 0.30 if self._style_pred_err > 0.30 else 0.45
        self.neural_weight += 0.05 * (target - self.neural_weight)
        self.neural_weight = clamp(self.neural_weight, 0.25, 0.80)

    def style_input(self, features: List[float], action: str,
                    valence: float, arousal: float) -> List[float]:
        """Features + strategy one-hot + mood, the style net's input."""
        inp = list(features) + [0.0] * len(ACTIONS) + [
            clamp(valence, -1.0, 1.0), clamp(arousal)]
        idx = ACTIONS.index(action) if action in ACTIONS else 0
        inp[len(features) + idx] = 1.0
        return inp

    def hybrid_score(self, features: List[float], action: str, state: str,
                     valence: float, arousal: float) -> float:
        """Hybrid strategy score: symbolic (Q + prior + context reward)
        blended with the style-conditioned neural prediction."""
        q = self.agent.q_value(state, action)
        context_reward = self.net.predict(features)[0] if features and any(features) else 0.5
        prior = ACTION_PRIORS.get(action, 0.5)
        symbolic = 0.4 * q + 0.3 * prior + 0.3 * context_reward
        neural = self.style_net.predict(
            self.style_input(features, action, valence, arousal))[0]
        return (1 - self.neural_weight) * symbolic + self.neural_weight * neural

    def state_for(self, ctx: CognitiveContext) -> str:
        novelty = float(ctx.scratch.get("novelty", 0.0))
        return QLearningAgent.state_key(ctx.has_question, ctx.emotions.valence, novelty)

    def choose_action(self, ctx: CognitiveContext) -> str:
        """Hybrid strategy selection: epsilon-greedy exploration over the
        blended symbolic + neural scores of every strategy."""
        state = self.state_for(ctx)
        features = self.encoder.encode(ctx.text) if ctx.text else [0.0] * self.encoder.dims
        valence = ctx.emotions.valence
        arousal = ctx.emotions.arousal

        scores = {a: self.hybrid_score(features, a, state, valence, arousal)
                  for a in ACTIONS}
        agent_rng = self.agent._rng
        if agent_rng.random() < self.agent.epsilon:
            action = agent_rng.choice(ACTIONS)
        else:
            action = max(ACTIONS, key=lambda a: scores[a])

        self._last_scores = scores
        self._pending = (state, action, features, valence, arousal)
        return action

    def feedback(self, reward: float) -> None:
        self._explicit_reward = max(-1.0, min(1.0, float(reward)))

    def credit_action(self, action: str) -> None:
        """Replaces the pending Q-credit's action — used when a module
        overrode the deliberated choice (e.g. the dual-process impulse
        answered before thinking). Keeps learning honest about what was
        actually spoken."""
        if self._pending is not None:
            state, _a, features, valence, arousal = self._pending
            self._pending = (state, action, features, valence, arousal)

    # ------------------------------------------------------------------ internals
    def _inferred_reward(self, text: str) -> float:
        cls, strength = sentiment_target(text)
        if cls == 2:
            return 0.3 * strength
        if cls == 0:
            return -0.4 * strength
        return 0.0

    def _replay_train(self, epochs: int = 3, lr: float = 0.15) -> None:
        items = list(self.replay)
        loss = self.net.train([x for x, _ in items], [t for _, t in items],
                              epochs=epochs, lr=lr, shuffle=True)
        self.train_batches += 1
        self.last_loss = loss

    def sentiment_of(self, text: str) -> Tuple[str, float]:
        """Predict sentiment class using the trained net."""
        if not text:
            return "neutral", 0.34
        probs = self.net.predict(self.encoder.encode(text))
        idx = max(range(3), key=lambda i: probs[i])
        return SENTIMENTS[idx], probs[idx]

    def stats(self) -> Dict[str, object]:
        return {"net_params": self.net.n_params,
                "epochs_trained": self.net.epochs_trained,
                "samples_seen": self.net.samples_seen,
                "replay_size": len(self.replay),
                "train_batches": self.train_batches,
                "last_loss": round(self.last_loss, 5),
                "q_learning": self.agent.stats(),
                "hybrid": {"neural_weight": round(self.neural_weight, 3),
                           "style_net_params": self.style_net.n_params,
                           "style_pred_err": round(self._style_pred_err, 3)},
                "vocabulary": self.lexicon.size}

    # ------------------------------------------------------------------ state
    def state_snapshot(self) -> dict:
        return {"net": self.net.state_dict(), "agent": self.agent.to_dict(),
                "style_net": self.style_net.state_dict(),
                "neural_weight": self.neural_weight,
                "lexicon": self.lexicon.to_dict(),
                "train_batches": self.train_batches, "last_loss": self.last_loss}

    def load_snapshot(self, data: dict) -> None:
        net_state = data["net"]
        if (net_state["n_in"], net_state["n_hidden"], net_state["n_out"]) != (
                self.net.n_in, self.net.n_hidden, self.net.n_out):
            # rebuild to the stored architecture (e.g. different encoder dims)
            self.net = MicroNet(net_state["n_in"], net_state["n_hidden"],
                                net_state["n_out"])
        self.net.load_state_dict(net_state)
        self.agent.load_dict(data["agent"])
        style_state = data.get("style_net")
        if style_state:
            if (style_state["n_in"], style_state["n_hidden"], style_state["n_out"]) != (
                    self.style_net.n_in, self.style_net.n_hidden, self.style_net.n_out):
                self.style_net = MicroNet(style_state["n_in"], style_state["n_hidden"],
                                          style_state["n_out"])
            self.style_net.load_state_dict(style_state)
        self.neural_weight = clamp(float(data.get("neural_weight", 0.45)), 0.25, 0.80)
        if data.get("lexicon"):
            self.lexicon.load_dict(data["lexicon"])
        self.train_batches = data.get("train_batches", 0)
        self.last_loss = data.get("last_loss", 0.0)
