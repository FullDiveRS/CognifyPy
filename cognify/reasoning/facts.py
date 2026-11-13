"""Facts and the fact parser: turning language into knowledge triples."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..util import singularize
_ids = itertools.count(1)

QUESTION_STARTERS = frozenset((
    "what", "who", "why", "how", "when", "where", "which", "is", "are", "can",
    "could", "does", "do", "did", "will", "would", "should", "tell",
))

# Clause openers that assert nothing: conditionals are hypothetical setups,
# existentials only announce a thing ("there is a 10 based number system").
NON_ASSERTIVE_STARTERS = frozenset(("if", "there is", "there are"))


def _collapse_enum(text: str) -> str:
    """"0, 1, 2" -> "0-1-2" — parity with the C# fact parser."""
    return re.sub(r",\s*", "-", text.strip())

PATTERNS: Tuple[Tuple[str, str], ...] = (
    # "joy means a bright feeling of gladness" — dictionary-style knowledge
    # (word, means, definition). Listed FIRST so the copula inside a
    # definition ("...that is coming") never hijacks the clause. Recallable
    # via "what does joy mean?" — parity with the C# VerbNormalization.
    (r"^(?:(?:a|an|the)\s+)?([a-z][\w\- ]*?)\s+means\s+([\w\- ]+?)\s*$", "means"),
    # "birds are not mammals" / "a bird is an animal" / "birds are animals"
    # / "the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9" (comma enumerations;
    # stored with the C# parity form "0-1-2-...")
    (r"^(?:(?:a|an|the)\s+)?([a-z][\w\- ]*?)\s+(are|is)\s+not\s+(?:(?:a|an|the)\s+)?([\w\-, ]+?)\s*$", "neg_isa"),
    (r"^(?:(?:a|an|the)\s+)?([a-z][\w\- ]*?)\s+(are|is)\s+(?:(?:a|an|the)\s+)?([\w\-, ]+?)\s*$", "isa"),
    # "birds cannot fly" / "birds can fly"  ("can be" is the copula of
    # possibility — "0 can be nothing" asserts no capability)
    (r"^([a-z][\w\- ]*?)\s+cannot\s+([\w\- ]+?)\s*$", "neg_can"),
    (r"^([a-z][\w\- ]*?)\s+can\s+([\w\- ]+?)\s*$", "can"),
    # "birds have wings" / "a bird has wings"
    (r"^(?:(?:a|an|the)\s+)?([a-z][\w\- ]*?)\s+(?:have|has)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s*$", "have"),
    # "rain causes floods" / "rain makes floods"
    (r"^([a-z][\w\- ]*?)\s+(?:causes?|makes?|leads? to|results? in)\s+([\w\- ]+?)\s*$", "causes"),
    # "sparrows eat seeds" / "cats like fish" / "cats hate water"
    (r"^(?:(?:a|an|the)\s+)?([a-z][\w\- ]*?)\s+(eats?|likes?|loves?|hates?|needs?|drinks?|avoids?)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s*$", "verb"),
)


@dataclass
class Fact:
    subject: str
    predicate: str                  # is-a | can | have | causes | is-not-a | <verb>
    object: str
    confidence: float = 0.9
    source: str = "user"
    chain: Tuple[str, ...] = ()     # derivation explanation (empty if user-taught)
    id: int = field(default_factory=lambda: next(_ids))

    def __post_init__(self) -> None:
        self.subject = singularize(self.subject.strip().lower())
        self.predicate = self.predicate.strip().lower()
        self.object = singularize(self.object.strip().lower())
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.chain = tuple(self.chain)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    @property
    def is_negated(self) -> bool:
        return self.predicate in ("is-not-a", "can-not")

    def explain(self) -> str:
        if not self.chain:
            return f"{self.subject} {self.predicate.replace('-', ' ')} {self.object}"
        steps = " and ".join(self.chain)
        return f"{self.subject} {self.predicate.replace('-', ' ')} {self.object} [because {steps}]"

    def to_dict(self) -> dict:
        return {"s": self.subject, "p": self.predicate, "o": self.object,
                "c": self.confidence, "src": self.source, "chain": list(self.chain)}

    @staticmethod
    def from_dict(d: dict) -> "Fact":
        return Fact(d["s"], d["p"], d["o"], d.get("c", .9), d.get("src", "user"),
                    tuple(d.get("chain", ())))

    def __str__(self) -> str:
        sym = {"is-a": "is a", "is-not-a": "is NOT a", "can": "can",
               "can-not": "cannot", "have": "has", "causes": "causes"}
        rel = sym.get(self.predicate, self.predicate)
        return f"{self.subject} {rel} {self.object}"


class FactParser:
    """Parses simple English statements into Facts."""

    def parse(self, text: str) -> List[Fact]:
        facts: List[Fact] = []
        if not text:
            return facts
        for clause in self._clauses(text):
            fact = self._parse_clause(clause)
            if fact is not None:
                facts.append(fact)
        return facts

    @staticmethod
    def _clauses(text: str) -> List[str]:
        """Split into clauses, carrying the subject across 'and can/have/...'."""
        text = text.strip().rstrip(".!?")
        out: List[str] = []
        verb_re = (r"(?:are|is|can(?:not)?|have|has|causes?|makes?|eats?|likes?|"
                   r"loves?|hates?|needs?|drinks?|avoids?)\b")
        subject_re = re.compile(
            r"^(?:(?:a|an|the)\s+)?([\w\- ]+?)\s+(?:" + verb_re + r")")
        for part in re.split(r"[;.]", text):
            part = part.strip()
            if not part:
                continue
            subject = ""
            for seg in re.split(r"\s+and\s+", part):
                seg = seg.strip()
                if not seg:
                    continue
                if re.match(r"^" + verb_re, seg):
                    if subject:
                        out.append(f"{subject} {seg}")
                    continue
                out.append(seg)
                m = subject_re.match(seg)
                if m:
                    subject = m.group(1).strip()
        return out

    _QUESTION_WORDS = {"what", "who", "when", "where", "why", "how", "which", "whose"}

    def _parse_clause(self, clause: str) -> Optional[Fact]:
        clause = clause.strip().lower()
        if not clause or len(clause.split()) > 16:
            return None
        # never extract "facts" from questions ("what causes damage?")
        if clause.split()[0] in QUESTION_STARTERS:
            return None
        # conditionals and existentials assert nothing about the world
        if any(clause.startswith(starter + " ") or clause == starter
               for starter in NON_ASSERTIVE_STARTERS):
            return None

        # Taught arithmetic ("2 + 2 is 4", "two plus two is four") is real
        # knowledge of a structured kind: ("2+2", equals, "4"). Refused
        # arithmetic (wrong result, e.g. "2 + 2 is 5") is a mistake, never a
        # generic fact about "2-2" — deferred import avoids a module cycle.
        from .arithmetic import try_teach, looks_like_arithmetic_statement
        arithmetic = try_teach(clause)
        if arithmetic is not None:
            return arithmetic
        if looks_like_arithmetic_statement(clause):
            return None

        for pattern, kind in PATTERNS:
            m = re.match(pattern, clause)
            if not m:
                continue
            groups = [g.strip() for g in m.groups() if g and g.strip()]
            # Unlabeled questions ("yume what is 2 + 2") must never become
            # facts like (yume-what, is-a, 2-2).
            words = set(re.findall(r"[a-z]+", clause))
            if words & self._QUESTION_WORDS:
                return None
            if kind == "neg_isa":
                return Fact(groups[0], "is-not-a", _collapse_enum(groups[-1]), 0.9)
            if kind == "isa":
                return Fact(groups[0], "is-a", _collapse_enum(groups[-1]), 0.9)
            if kind == "means":
                # C# parity: articles are stripped from the whole object
                # phrase ("a bright feeling of a hug" -> "bright feeling of
                # hug"), keeping stored meanings identical across both minds.
                obj = re.sub(r"\b(?:a|an|the)\s+", "", f" {groups[-1]} ").strip()
                return Fact(groups[0], "means", obj, 0.9)
            if kind == "neg_can":
                return Fact(groups[0], "can-not", groups[-1], 0.9)
            if kind == "can":
                return Fact(groups[0], "can", groups[-1], 0.9)
            if kind == "have":
                return Fact(groups[0], "have", groups[-1], 0.85)
            if kind == "causes":
                return Fact(groups[0], "causes", groups[-1], 0.85)
            if kind == "verb":
                verb = singularize(groups[1].rstrip("s") if groups[1].endswith("s") else groups[1])
                return Fact(groups[0], verb, groups[-1], 0.85)
        return None
