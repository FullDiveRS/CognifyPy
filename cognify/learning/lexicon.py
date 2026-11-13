"""Lexicon: the mind's growing word vocabulary.

Every cognitive cycle the learner listens to what the user says and picks up
words it has never heard before; each new word is remembered with the cycle
it appeared in. The composer weaves freshly picked words back into
conversation ("I just picked up the word ... — what does it mean to you?"),
so every new word the user uses becomes one more thing the mind can say —
steadily widening its replies beyond the built-in templates and directly
fighting the repetition problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..util import tokenize

#: Common function words that never enter the vocabulary.
STOP_WORDS = {
    "what", "who", "how", "why", "when", "where", "which", "the", "a", "an",
    "is", "are", "was", "were", "do", "does", "did", "you", "your", "yours",
    "i", "me", "my", "mine", "we", "us", "our", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "about", "tell", "can", "could", "would", "will", "shall",
    "should", "let", "think", "know", "like", "want", "have", "has", "had",
    "be", "been", "am", "if", "so", "not", "no", "yes", "at", "as", "by",
    "from", "up", "out", "get", "got", "go", "going", "some", "any", "yet",
}

#: Time/degree words never make good vocabulary picks — weaving "I just
#: picked up the word 'currently'" into a reply would sound like a glitch.
#: Mirrors the topic-filler list the response composer already skips.
#: Extended after real conversations showed the mind "picking up" speech
#: glue: okay, getting, doing, good, job, lets, try, slightly, vocab, fore.
FILLER_WORDS = {
    "currently", "now", "then", "today", "recently", "still", "already",
    "again", "always", "never", "maybe", "perhaps", "really", "actually",
    "just", "very", "many", "much", "also", "too", "quite", "pretty",
    # speech glue and semi-words heard in real chat
    "okay", "ok", "good", "fine", "well", "getting", "doing", "lets",
    "let", "try", "trying", "tried", "job", "vocab", "fore", "slightly",
    "thing", "things", "stuff", "gonna", "wanna", "gotta", "kinda",
    "sorta", "yeah", "yep", "nah", "hmm", "huh", "wow", "um", "uh",
    "like", "lot", "lots", "bit", "way", "sure", "please",
}


@dataclass
class Lexeme:
    """One known word and where it came from."""
    word: str
    count: int = 1
    first_cycle: int = 0
    last_cycle: int = 0


class Lexicon:
    """Words picked up from the user while talking, plus which of them have
    already been woven into a reply (a word is asked about at most once)."""

    def __init__(self) -> None:
        self._words: Dict[str, Lexeme] = {}
        self._recent: List[str] = []          # newest last
        self._spoken: set = set()             # already woven into a reply
        self._last_picks: List[str] = []

    # ------------------------------------------------------------------ query
    @property
    def size(self) -> int:
        return len(self._words)

    @property
    def last_picks(self) -> List[str]:
        """Words picked up during the LAST observed cycle (what the demo
        shows as "she just picked up ...")."""
        return list(self._last_picks)

    def all(self) -> List[Lexeme]:
        return sorted(self._words.values(), key=lambda w: (w.first_cycle, w.word))

    def recent_new_words(self, max_n: int = 8) -> List[str]:
        """The most recently picked words, newest first."""
        return list(reversed(self._recent))[:max_n]

    def fresh_word(self) -> Optional[str]:
        """The freshest not-yet-spoken word for the composer, or None.
        Peeking does not consume — mark_spoken does that once the reply
        wins deliberation."""
        for word in reversed(self._recent):
            if word not in self._spoken:
                return word
        return None

    def mark_spoken(self, word: str) -> None:
        """Marks a word as having been woven into a reply (consumes it)."""
        self._spoken.add((word or "").strip().lower())

    def was_spoken(self, word: str) -> bool:
        return (word or "").strip().lower() in self._spoken

    def knows(self, word: str) -> bool:
        return (word or "").strip().lower() in self._words

    # ------------------------------------------------------------------ cycle
    def observe(self, text: str, cycle: int) -> List[str]:
        """Listens to one utterance, registering every content word (3+
        letters, no stopwords) never heard before. Returns the words picked
        up THIS cycle, in the order they appeared."""
        picked: List[str] = []
        if not text:
            self._last_picks = []
            return picked
        for token in tokenize(text):
            if len(token) < 3 or token in STOP_WORDS or token in FILLER_WORDS:
                continue
            lexeme = self._words.get(token)
            if lexeme is not None:
                lexeme.count += 1
                lexeme.last_cycle = cycle
                continue
            self._words[token] = Lexeme(word=token, count=1, first_cycle=cycle, last_cycle=cycle)
            self._recent.append(token)
            if len(self._recent) > 16:
                self._recent.pop(0)
            picked.append(token)
        self._last_picks = list(picked)
        return picked

    # ------------------------------------------------------------------ state
    def to_dict(self) -> dict:
        return {
            "words": [[w.word, w.count, w.first_cycle, w.last_cycle]
                      for w in self.all()],
            "spoken": sorted(self._spoken),
        }

    def load_dict(self, data: dict) -> None:
        self._words.clear()
        self._recent.clear()
        self._spoken.clear()
        self._last_picks = []
        for row in (data or {}).get("words", []):
            try:
                word, count, first, last = row[0], int(row[1]), int(row[2]), int(row[3])
            except (ValueError, IndexError, TypeError):
                continue
            if not word:
                continue
            self._words[word] = Lexeme(word=word, count=count, first_cycle=first, last_cycle=last)
            self._recent.append(word)
        if len(self._recent) > 16:
            self._recent = self._recent[-16:]
        for word in (data or {}).get("spoken", []):
            if word:
                self._spoken.add(word)
