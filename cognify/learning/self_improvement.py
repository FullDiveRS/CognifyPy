"""SelfImprovement: the mind studying its own conversations and improving.

Mirrors the C# Cognify.Learning.SelfImprovementModule — three mechanisms:

1. IMPLIED FEEDBACK: the user's next message often grades the previous reply
   ("good job!", "no, that's wrong", "stop saying that"). Those cues become
   real reward for the hybrid Q-engine, attributed to the action that was
   actually spoken one turn earlier.
2. WORDING SCORES: every reply template keeps an approval EMA. Disliked
   wordings are demoted (deliberation steers away), approved ones promoted;
   a self-review retires trained sayings whose score stays negative.
3. SELF-REVIEW: ``self_improve()`` runs an explainable pass over the ledger.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# Cues that grade the previous reply. Deliberately conservative — a false
# "dislike" would suppress good wordings.
POSITIVE_CUES = (
    "good job", "well done", "nice work", "that's right", "thats right",
    "that is right", "correct", "exactly", "perfect", "great answer",
    "good answer", "smart", "you got it", "that helped", "thanks",
    "thank you", "i like that", "love that", "amazing", "awesome",
    "you're smart", "you are smart", "clever", "proud of you",
)

NEGATIVE_CUES = (
    "wrong", "incorrect", "not right", "that's not", "thats not",
    "that is not", "you don't understand", "you do not understand",
    "you're not listening", "you are not listening", "stop saying",
    "stop talking like", "bad answer", "nonsense", "that made no sense",
    "you're repeating", "you are repeating", "already said that",
    "stupid", "dumb", "idiot", "shut up", "annoying", "boring",
    "you misheard", "you misunderstood", "no, that", "no that",
)

_RETIRE_THRESHOLD = -0.45


def classify_feedback(text: Optional[str]) -> int:
    """+1 approval of the previous reply, -1 disapproval, 0 no cue.
    Questions are questions, not grades."""
    if not text or not text.strip():
        return 0
    lower = text.lower()
    if "?" in lower:
        return 0
    for cue in NEGATIVE_CUES:
        if cue in lower:
            return -1
    for cue in POSITIVE_CUES:
        if cue in lower:
            return 1
    return 0


def _template_key(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", text.strip().lower())


class SelfImprovementReport:
    """What one self-improvement review pass changed."""

    def __init__(self, *, cycles_observed: int = 0, positives: int = 0,
                 negatives: int = 0, templates_tracked: int = 0,
                 templates_demoted: int = 0, templates_promoted: int = 0,
                 sayings_pruned: int = 0, summary: str = "") -> None:
        self.cycles_observed = cycles_observed
        self.positives = positives
        self.negatives = negatives
        self.templates_tracked = templates_tracked
        self.templates_demoted = templates_demoted
        self.templates_promoted = templates_promoted
        self.sayings_pruned = sayings_pruned
        self.summary = summary

    def __str__(self) -> str:  # pragma: no cover
        return self.summary

    def __repr__(self) -> str:  # pragma: no cover
        return f"SelfImprovementReport({self.summary})"


class SelfImprovement:
    """Feedback ledger + wording reputations for one mind."""

    def __init__(self) -> None:
        self._template_scores: Dict[str, float] = {}
        self._cycles = 0
        self._positives = 0
        self._negatives = 0
        self._pruned_sayings = 0
        # Snapshot of the previous turn: (cycle, action, template, learner pending)
        self._prev: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ stats
    @property
    def cycles_observed(self) -> int:
        return self._cycles

    @property
    def positives(self) -> int:
        return self._positives

    @property
    def negatives(self) -> int:
        return self._negatives

    # ------------------------------------------------------------------ cycle
    def observe_turn(self, mind, text: str, response, cycle: int) -> None:
        """Called at the end of every cycle: grade the previous turn when this
        turn's text carries implied feedback, then snapshot this turn."""
        learner = getattr(mind, "learner", None)
        cue = classify_feedback(text)
        if cue != 0 and self._prev is not None and self._prev["cycle"] == cycle - 1:
            self._apply(mind, learner, cue)

        pending = getattr(learner, "_pending", None) if learner is not None else None
        self._prev = {
            "cycle": cycle,
            "action": response.meta.get("action") if response is not None else None,
            "template": _template_key(response.text) if response is not None else None,
            "pending": tuple(pending) if pending is not None else None,
        }
        self._cycles += 1

    def _apply(self, mind, learner, cue: int) -> None:
        reward = 0.65 if cue > 0 else -0.75
        template = self._prev["template"]
        if template:
            current = self._template_scores.get(template, 0.0)
            current += 0.35 * (reward - current)
            self._template_scores[template] = max(-1.0, min(1.0, current))
        if cue > 0:
            self._positives += 1
        else:
            self._negatives += 1

        # Close the reinforcement loop on the action that was actually
        # spoken: hand the hybrid engine the graded turn's pending tuple and
        # the reward; the next observe() applies the Q update + style net.
        if learner is not None and self._prev["pending"] is not None:
            learner._pending = self._prev["pending"]
            learner._explicit_reward = reward

    # ------------------------------------------------------------------ scores
    def template_penalty(self, template_key: str) -> float:
        """0..1 penalty for a wording that historically drew disapproval."""
        s = self._template_scores.get(template_key, 0.0)
        return min(1.0, -s) if s < 0 else 0.0

    def template_boost(self, template_key: str) -> float:
        """0..1 boost for a wording that historically drew approval."""
        s = self._template_scores.get(template_key, 0.0)
        return min(1.0, s) if s > 0 else 0.0

    def record_feedback(self, reply_text: str, reward: float) -> None:
        """Grades a specific spoken reply with an explicit reward in -1..1
        (UI thumbs-up, evaluation harness). Moves the wording's reputation
        exactly like implied feedback does."""
        if not reply_text:
            return
        reward = max(-1.0, min(1.0, float(reward)))
        template = _template_key(reply_text)
        current = self._template_scores.get(template, 0.0)
        current += 0.35 * (reward - current)
        self._template_scores[template] = max(-1.0, min(1.0, current))
        if reward >= 0:
            self._positives += 1
        else:
            self._negatives += 1

    # ------------------------------------------------------------------ review
    def self_improve(self, mind) -> SelfImprovementReport:
        """One conscious review: retire sayings that never land, summarize."""
        pruned = 0
        for saying in list(mind.phrases.all()):
            if self._template_scores.get(_template_key(saying), 0.0) <= _RETIRE_THRESHOLD:
                if mind.phrases.remove(saying):
                    pruned += 1
                    self._pruned_sayings += 1
        demoted = sum(1 for s in self._template_scores.values() if s <= _RETIRE_THRESHOLD)
        promoted = sum(1 for s in self._template_scores.values() if s >= 0.45)
        summary = (
            f"Reviewed {self._cycles} turn(s) — {self._positives} approval(s), "
            f"{self._negatives} disapproval(s); {len(self._template_scores)} wording(s) tracked "
            f"({promoted} promoted, {demoted} demoted), {pruned} saying(s) retired this pass.")
        return SelfImprovementReport(
            cycles_observed=self._cycles, positives=self._positives,
            negatives=self._negatives, templates_tracked=len(self._template_scores),
            templates_demoted=demoted, templates_promoted=promoted,
            sayings_pruned=pruned, summary=summary)

    # ------------------------------------------------------------------ state
    def to_dict(self) -> dict:
        return {
            "templates": [[k, v] for k, v in
                          sorted(self._template_scores.items(),
                                 key=lambda kv: -abs(kv[1]))[:300]],
            "cycles": self._cycles,
            "positives": self._positives,
            "negatives": self._negatives,
            "pruned_sayings": self._pruned_sayings,
        }

    def load_dict(self, data: dict) -> None:
        if not data:
            return
        self._template_scores = {
            row[0]: max(-1.0, min(1.0, float(row[1])))
            for row in data.get("templates", []) if len(row) == 2
        }
        self._cycles = int(data.get("cycles", 0))
        self._positives = int(data.get("positives", 0))
        self._negatives = int(data.get("negatives", 0))
        self._pruned_sayings = int(data.get("pruned_sayings", 0))
        self._prev = None
