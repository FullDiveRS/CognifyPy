"""InferenceEngine: forward chaining, induction, abduction, and query answering."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..core.context import ReasoningAnswer
from ..util import singularize, tokenize
from .facts import Fact
from .knowledge import KnowledgeBase


class InferenceEngine:
    """Derives new facts and answers questions about the knowledge base."""

    MAX_FACTS = 500

    def __init__(self) -> None:
        self.last_derivation: Tuple[str, ...] = ()

    # ------------------------------------------------------------------ rules
    def forward_chain(self, kb: KnowledgeBase, max_rounds: int = 6) -> List[Fact]:
        """Run all rules to fixpoint; return newly derived facts."""
        derived: List[Fact] = []
        seen_keys = {f.key for f in kb.facts()}
        for _ in range(max_rounds):
            if kb.size() >= self.MAX_FACTS:
                break
            fresh: List[Fact] = []
            fresh += self._rule_transitive_isa(kb, seen_keys)
            fresh += self._rule_property_inheritance(kb, seen_keys)
            fresh += self._rule_causal_chain(kb, seen_keys)
            if not fresh:
                break
            for f in fresh:
                if kb.add(f):
                    derived.append(f)
                    seen_keys.add(f.key)
        return derived

    def _push(self, out: List[Fact], seen: set, fact: Fact) -> None:
        if fact.key not in seen and fact.subject != fact.object:
            out.append(fact)

    def _rule_transitive_isa(self, kb: KnowledgeBase, seen: set) -> List[Fact]:
        out: List[Fact] = []
        for f in kb.facts():
            if f.predicate != "is-a" or f.is_negated:
                continue
            for g in kb.about(f.object, "is-a"):
                if g.is_negated:
                    continue
                conf = min(f.confidence, g.confidence) * 0.9
                chain = (f"{f.subject} is a {f.object}",
                         f"{f.object} is a {g.object}")
                self._push(out, seen, Fact(f.subject, "is-a", g.object, conf,
                                           "inference", chain))
        return out

    def _rule_property_inheritance(self, kb: KnowledgeBase, seen: set) -> List[Fact]:
        out: List[Fact] = []
        for f in kb.facts():
            if f.predicate != "is-a" or f.is_negated:
                continue
            for g in kb.about(f.object):
                if g.predicate in ("is-a", "is-not-a") or g.is_negated:
                    continue
                conf = min(f.confidence, g.confidence) * 0.9
                chain = (f"{f.subject} is a {f.object}",
                         f"{f.object} {g.predicate} {g.object}")
                self._push(out, seen, Fact(f.subject, g.predicate, g.object,
                                           conf, "inference", chain))
        return out

    def _rule_causal_chain(self, kb: KnowledgeBase, seen: set) -> List[Fact]:
        out: List[Fact] = []
        for f in kb.facts():
            if f.predicate != "causes":
                continue
            for g in kb.about(f.object, "causes"):
                conf = min(f.confidence, g.confidence) * 0.85
                chain = (f"{f.subject} causes {f.object}",
                         f"{f.object} causes {g.object}")
                self._push(out, seen, Fact(f.subject, "causes", g.object, conf,
                                           "inference", chain))
        return out

    def induce(self, kb: KnowledgeBase, min_instances: int = 3) -> List[Fact]:
        """Induction: if >= min_instances of a class can X, generalize."""
        derived: List[Fact] = []
        classes: dict = {}
        for f in kb.facts():
            if f.predicate != "is-a" or f.is_negated:
                continue
            classes.setdefault(f.object, set()).add(f.subject)
        for cls, members in classes.items():
            for pred in ("can",):
                positive = {m for m in members
                            if kb.get(m, pred, "") is not None or any(
                                f.subject == m and f.predicate == pred
                                for f in kb.facts())}
                for prop_fact in kb.facts():
                    if prop_fact.predicate != pred or prop_fact.is_negated:
                        continue
                    if prop_fact.subject in members:
                        positive.add(prop_fact.subject)
                negative = {m for m in members
                            if kb.get(m, "can-not", prop_fact.object) is not None}
                support = {m for m in positive
                           if any(f.subject == m and f.predicate == pred
                                  and f.object == prop_fact.object
                                  for f in kb.facts())}
                if len(support) >= min_instances and not negative:
                    key = (cls, pred, prop_fact.object)
                    if key not in {(f.subject, f.predicate, f.object) for f in kb.facts()}:
                        conf = min(0.75, 0.45 + 0.08 * len(support))
                        chain = (f"{len(support)} members of {cls} can {prop_fact.object}",)
                        fact = Fact(cls, pred, prop_fact.object, conf, "induction", chain)
                        if kb.add(fact):
                            derived.append(fact)
        return derived

    def abduce(self, kb: KnowledgeBase, effect: str) -> List[Fact]:
        """Abduction: hypothesize causes for an observed effect."""
        effect = singularize((effect or "").strip().lower())
        hyps: List[Fact] = []
        for f in kb.facts():
            if f.predicate == "causes" and f.object == effect:
                chain = (f"{f.subject} causes {effect}",)
                hyps.append(Fact(f"{f.subject}(hypothetical)", "explains", effect,
                                 f.confidence * 0.6, "abduction", chain))
        return hyps

    # ------------------------------------------------------------------ queries
    # Preambles must not hide the question: "now what is after?" is a
    # question about "after" — leading filler words are dropped before
    # matching (parity with the C# AnswerQuestion leading-filler skip).
    _LEADING_FILLERS = re.compile(
        r"^(?:now|then|also|so|well|okay|ok|and|plus|today|please|anyway|"
        r"just|still|simply)\s+", re.I)

    QUERY_DESCRIBE = re.compile(
        r"^(?:what|who)\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s*\??$", re.I)
    QUERY_MEANS = re.compile(
        r"^what\s+(?:does|do)\s+([\w\- ]+?)\s+mean\s*\??$", re.I)
    QUERY_MEANING_OF = re.compile(
        r"^what\s+is\s+the\s+meaning\s+of\s+([\w\- ]+?)\s*\??$", re.I)
    QUERY_DEFINE = re.compile(
        r"^define\s+([\w\- ]+?)\s*\??$", re.I)
    QUERY_YESNO_ISA = re.compile(
        r"^(?:is|are)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s*\??$", re.I)
    QUERY_CAN = re.compile(
        r"^(?:can|does|do)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s*\??$", re.I)
    QUERY_WHY = re.compile(
        r"^why\s+(?:can|do|does|is|are)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)\s*\??$", re.I)
    QUERY_CAUSES = re.compile(
        r"^(?:what|why)\s+(?:causes?|does)\s+(?:(?:a|an|the)\s+)?([\w\- ]+?)(?:\s+happen|\s+occur)?\s*\??$", re.I)

    def answer(self, kb: KnowledgeBase, text: str) -> Optional[ReasoningAnswer]:
        """Answer a natural-language question from the knowledge base."""
        t = (text or "").strip().rstrip("?").strip()
        if not t:
            return None
        # Drop preamble fillers: "now what is after?" -> "what is after?"
        while True:
            stripped = self._LEADING_FILLERS.sub("", t, count=1)
            if stripped == t:
                break
            t = stripped

        m = self.QUERY_WHY.match(t)
        if m:
            return self._answer_why(kb, m.group(1), m.group(2))

        # "what does X mean?" / "what is the meaning of X?" / "define X" —
        # dictionary-style lookups over the taught means facts. Checked
        # BEFORE QUERY_DESCRIBE (whose "what is X" shape would otherwise
        # swallow "what is the meaning of X" into a lookup for a concept
        # named "meaning of X") and before QUERY_CAUSES, which would
        # misread "what does happy mean" as a question about causes.
        m = self.QUERY_MEANS.match(t)
        if m:
            return self._answer_means(kb, m.group(1))
        m = self.QUERY_MEANING_OF.match(t)
        if m:
            return self._answer_means(kb, m.group(1))
        m = self.QUERY_DEFINE.match(t)
        if m:
            return self._answer_means(kb, m.group(1))

        m = self.QUERY_DESCRIBE.match(t)
        if m:
            return self._answer_describe(kb, m.group(1))

        m = self.QUERY_CAUSES.match(t)
        if m and not m.group(1).strip().lower().endswith("mean"):
            return self._answer_causes(kb, m.group(1))

        m = self.QUERY_YESNO_ISA.match(t)
        if m and re.match(r"^is\b", t.lower()):
            return self._answer_yesno(kb, m.group(1), "is-a", m.group(2))

        m = self.QUERY_CAN.match(t)
        if m and re.match(r"^can\b", t.lower()):
            return self._answer_yesno(kb, m.group(1), "can", m.group(2))
        return None

    @staticmethod
    def _with_article(word: str) -> str:
        return ("an " if word[:1] in "aeiou" else "a ") + word

    @staticmethod
    def _plural(word: str) -> str:
        if re.search(r"(s|x|z|ch|sh)$", word):
            return word + "es"
        if word.endswith("y") and word[-2:-1] not in "aeiou":
            return word[:-1] + "ies"
        return word + "s"

    # Conversational fillers that ride on the "what does ___ mean" shape:
    # "what do you mean?" is not a dictionary lookup.
    _MEANS_STOPWORDS = frozenset((
        "you", "i", "me", "it", "this", "that", "these", "those",
        "they", "them", "we", "us", "he", "she", "everything",
        "anything", "something", "nothing"))

    def _answer_means(self, kb: KnowledgeBase, word: str) -> Optional[ReasoningAnswer]:
        """Answer "what does X mean?" from taught means facts — parity with
        the C# DescribeEntity route over "what does X ..." questions."""
        w = (word or "").strip().lower()
        if not w or w in self._MEANS_STOPWORDS:
            return None
        ans = self._answer_describe(kb, w)
        if ans is not None:
            return ans
        return ReasoningAnswer(
            f"I don't know what {singularize(w)} means yet — teach me!", 0.2, "unknown")

    def _answer_describe(self, kb: KnowledgeBase, concept: str) -> Optional[ReasoningAnswer]:
        concept = concept.strip().lower()
        parts: List[str] = []
        confs: List[float] = []
        display = concept if kb.about(concept) else singularize(concept)
        # The definition comes first — for a word the mind was taught,
        # "joy means ..." IS the answer to "what is joy?".
        for f in (kb.about(concept, "means") or kb.about(singularize(concept), "means")):
            parts.append(f"{display} means {f.object}")
            confs.append(f.confidence)
        facts = kb.about(concept, "is-a") or kb.about(singularize(concept), "is-a")
        for f in facts:
            if display.endswith("s"):
                parts.append(f"{display} are {self._plural(f.object)}")
            else:
                parts.append(f"{self._with_article(display).capitalize()} is {self._with_article(f.object)}")
            confs.append(f.confidence)
        lookup = display
        for pred, phrasing in (("can", "it can"), ("have", "it has")):
            seen_props = set()
            props = kb.inherited(lookup, pred) or kb.inherited(singularize(lookup), pred)
            for f, conf in props:
                if f.object in seen_props:
                    continue
                seen_props.add(f.object)
                obj = self._plural(f.object) if pred == "have" else f.object
                parts.append(f"{phrasing} {obj}")
                confs.append(conf)
        if not parts:
            return None
        sentences = [p[0].upper() + p[1:] for p in parts]
        text = ". ".join(sentences) + "."
        conf = sum(confs) / len(confs)
        return ReasoningAnswer(text=text, confidence=conf, kind="describe")

    def _answer_yesno(self, kb: KnowledgeBase, subject: str, pred: str, obj: str) -> ReasoningAnswer:
        subject, obj = singularize(subject.strip().lower()), singularize(obj.strip().lower())
        neg = kb.get(subject, "is-not-a" if pred == "is-a" else "can-not", obj)
        if neg is not None:
            return ReasoningAnswer(f"No — {subject} is explicitly not {obj} "
                                   f"(you taught me that).", 0.9, "yesno")
        direct = kb.get(subject, pred, obj)
        if direct is not None:
            return ReasoningAnswer(f"Yes — {direct.explain()}.", direct.confidence, "yesno")
        # inheritance path
        if pred == "is-a":
            path = kb.has_isa_path(subject, obj)
            if path:
                chain = tuple(f"{path[i]} is a {path[i+1]}" for i in range(len(path) - 1))
                return ReasoningAnswer(
                    f"Yes — {' and '.join(chain)}, so {subject} is a {obj}.",
                    0.85, "yesno", chain)
        else:
            inherited = kb.inherited(subject, pred)
            for f, conf in inherited:
                if f.object == obj:
                    return ReasoningAnswer(
                        f"Yes — {f.explain()} (inherited through the class hierarchy).",
                        conf, "yesno", f.chain)
        about = kb.about(subject)
        if about:
            return ReasoningAnswer(
                f"I don't know whether {subject} {pred} {obj} yet. "
                f"I do know {len(about)} thing(s) about {subject}.", 0.3, "yesno")
        return ReasoningAnswer(f"I don't know what {subject} is yet — teach me!", 0.2, "unknown")

    def _answer_why(self, kb: KnowledgeBase, subject: str, obj: str) -> ReasoningAnswer:
        subject = singularize(subject.strip().lower())
        obj = singularize(obj.strip().lower())
        fact = kb.get(subject, "can", obj) or kb.get(subject, "is-a", obj)
        if fact is None:
            return ReasoningAnswer(f"I have no recorded reason why {subject} {obj}.", 0.2, "unknown")
        if fact.chain:
            text = f"{subject} {obj}: " + " and ".join(fact.chain) + "."
            return ReasoningAnswer(text, fact.confidence, "why", fact.chain)
        # try to reconstruct an is-a explanation
        if fact.predicate == "can":
            for parent in kb.parents(subject):
                if kb.get(parent, "can", obj) is not None:
                    text = (f"{subject} can {obj} because {subject} are {parent} "
                            f"and {parent} can {obj}.")
                    return ReasoningAnswer(text, min(fact.confidence, .85), "why",
                                           (f"{subject} is a {parent}",
                                            f"{parent} can {obj}"))
        return ReasoningAnswer(f"{fact.explain()} — that's what I was taught.",
                               fact.confidence, "why")

    def _answer_causes(self, kb: KnowledgeBase, effect: str) -> ReasoningAnswer:
        effect = singularize(effect.strip().lower())
        causes: List[str] = []
        chains: List[str] = []
        for f in kb.facts():
            if f.predicate == "causes" and f.object == effect:
                causes.append(f.subject)
                if f.chain:
                    chains.append(" via " + " and ".join(f.chain))
        if not causes:
            hyps = self.abduce(kb, effect)
            if hyps:
                return ReasoningAnswer(
                    f"I'm not certain, but perhaps {hyps[0].subject.replace('(hypothetical)', '')} "
                    f"— since it causes {effect}.", hyps[0].confidence, "causes")
            return ReasoningAnswer(f"I don't know what causes {effect} yet.", 0.2, "unknown")
        text = f"Known causes of {effect}: " + ", ".join(causes) + "." + "".join(chains[:1])
        return ReasoningAnswer(text, 0.8, "causes")
