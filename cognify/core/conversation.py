"""The conversation tracker — the mind's running UNDERSTANDING of the chat.

Mirrors the C# ``ConversationTracker``: which topics keep coming back,
which of the user's questions were never fully answered, and how familiar
the two of you have become. Deliberately lightweight — a host feeds each
turn to :meth:`ConversationTracker.observe`, the composer may reference the
result ("Earlier you mentioned …"), and every few turns the top topics flow
into long-term memory so the understanding survives a save/reload.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..util import clamp, tokenize

# Words that never make a good conversation topic (mirror of the composer's
# fillers — the tracker runs before/without the composer in some paths).
_FILLERS = {
    "the", "and", "but", "for", "with", "about", "that", "this", "have",
    "has", "had", "was", "were", "are", "did", "will", "would", "can",
    "could", "should", "you", "your", "she", "her", "him", "his", "not",
    "what", "who", "when", "where", "why", "how", "currently", "now",
    "then", "today", "still", "already", "again", "just", "very", "many",
    "much", "also", "too", "really", "actually", "like", "know", "think",
}


class ConversationTracker:
    """Collects recurring topics, open questions and familiarity."""

    def __init__(self) -> None:
        self.turns = 0
        self._topic_counts: Dict[str, int] = {}
        self._pending: List[Tuple[str, int]] = []
        self._last_memory_turn = 0

    # ------------------------------------------------------------------ state
    @property
    def familiarity(self) -> float:
        """0..1 — how familiar the conversation has become (turns/20, capped)."""
        return min(1.0, self.turns / 20.0)

    @property
    def topics(self) -> List[str]:
        """The conversation's recurring topics, most frequent first."""
        return [t for t, _ in sorted(self._topic_counts.items(),
                                     key=lambda kv: (-kv[1], kv[0]))[:12]]

    @property
    def pending_questions(self) -> List[str]:
        """Questions the user asked that the mind's replies never clearly addressed."""
        return [q for q, _ in self._pending]

    # ------------------------------------------------------------------ observe
    def observe(self, mind, user_text: str, reply_text: str, cycle: int) -> None:
        """Observes one turn (user text + the mind's reply). Topics are
        counted from the user's content words; a user question whose reply
        shares none of its content words is kept as an open thread (max 3,
        forgotten after 8 more turns). Every 6th turn the top topics land in
        long-term memory."""
        if not user_text or not user_text.strip():
            return
        self.turns += 1

        for word in tokenize(user_text):
            if len(word) > 2 and word not in _FILLERS:
                self._topic_counts[word] = self._topic_counts.get(word, 0) + 1
        while len(self._topic_counts) > 40:
            rarest = min(self._topic_counts.items(), key=lambda kv: kv[1])[0]
            del self._topic_counts[rarest]

        if "?" in user_text:
            reply_words = {w.lower() for w in tokenize(reply_text or "")}
            addressed = any(w.lower() in reply_words
                            for w in tokenize(user_text)
                            if len(w) > 3 and w not in _FILLERS)
            if not addressed:
                self._pending.append((user_text.strip()[:120], cycle))
                if len(self._pending) > 3:
                    self._pending.pop(0)
        self._pending = [(q, c) for q, c in self._pending if cycle - c <= 8]

        # Persistence: every 6th turn, the conversation's shape lands in
        # long-term memory — how understanding survives save/reload. The
        # "conversation" tag mirrors the C# tracker's memory source.
        if self.turns - self._last_memory_turn >= 6:
            self._last_memory_turn = self.turns
            top = self.topics
            if top and mind is not None:
                mind.memory.add(
                    f"We have been talking about {', '.join(top[:2])}.",
                    kind="fact", importance=clamp(0.45), tags={"conversation"})

    # ------------------------------------------------------------------ reads
    def summary(self) -> List[str]:
        """A short developer-facing summary of what the mind understands
        about this conversation right now."""
        lines = [f"we have exchanged {self.turns} message(s) "
                 f"(familiarity {self.familiarity:.0%})"]
        top = self.topics[:5]
        lines.append(f"recurring topics: {', '.join(top)}" if top
                     else "no recurring topics yet")
        if self._pending:
            lines.append("still open: " + " / ".join(q for q, _ in self._pending))
        return lines

    def latest_topic(self) -> str:
        """The topic to reference when a reply wants to nod at the shared
        thread, or ``""``. Ties break alphabetically (same order as
        :attr:`topics`), so the choice is stable."""
        if not self._topic_counts:
            return ""
        return sorted(self._topic_counts.items(),
                      key=lambda kv: (-kv[1], kv[0]))[0][0]

    def has_open_thread(self) -> bool:
        return bool(self._pending)

    def oldest_open_question(self) -> Optional[str]:
        return self._pending[0][0] if self._pending else None
