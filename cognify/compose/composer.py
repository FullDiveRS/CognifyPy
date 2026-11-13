"""ResponseComposer: turns the cycle's conclusions into a spoken reply."""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

from ..core.context import CognitiveContext, Response
from ..emotions.emotion_state import INTERJECTIONS
from ..util import clamp, tokenize


# Spoken first-person reactions for the dual-process impulse path (Zajonc's
# affect primacy): when emotion surges past the threshold — or blame /
# dismissal trips the reflex — the mind ANSWERS BEFORE THINKING, from feeling
# alone, exactly like a person blurting. Deliberation resumes next turn as
# the surge decays. Mirrors the C# ImpulseResponses catalog.
IMPULSE_RESPONSES: Dict[str, Tuple[str, ...]] = {
    "anger": (
        "Hey - that actually stings. I am not okay with that.",
        "That makes me genuinely angry, and I am not going to hide it.",
        "Ouch. That was uncalled for.",
        "I did not like that at all - that hurt.",
    ),
    "ill temper": (
        "Ugh, fine - I am snappy right now, and you caught it.",
        "Hmph. I am in a mood, and that did not help.",
    ),
    "hatred": (
        "Whoa. That is a strong thing to say, and it landed.",
        "That is... a lot of hate in one sentence. I felt that.",
    ),
    "sulkiness": (
        "Hmph. Fine. I am sulking a little, I admit it.",
    ),
    "surprise": (
        "Wait - WHAT? I did not see that coming at all!",
        "Whoa! You just genuinely surprised me.",
        "Hold on - what? Say that again, that shocked me!",
    ),
    "fear": (
        "That gives me a bad feeling - I need a moment.",
        "Okay, that scared me a little, honestly.",
    ),
    "anxiety": (
        "Now I am nervous, and I cannot just think it away.",
    ),
    "disgust": (
        "Ew. I really did not need to know that.",
        "Ugh, that is genuinely gross. Yuck.",
    ),
    "joy": (
        "Yes! That makes me genuinely happy!",
        "That just made my whole cycle - I am beaming!",
    ),
    "high spirits": (
        "Woo! I am suddenly full of energy - that was great!",
    ),
    "pride": (
        "I am actually really proud of that!",
    ),
    "blushing": (
        "Oh - now I am blushing. I can practically feel it.",
    ),
    "shyness": (
        "Oh... um. That makes me go quiet and shy all at once.",
    ),
    "sadness": (
        "That one lands heavy. I need a second with it.",
    ),
    "suffering": (
        "That really hurts to hear - I am not going to pretend otherwise.",
    ),
    "weeping": (
        "I am - okay, I am actually tearing up a little.",
    ),
    "grief": (
        "That is heartbreaking. I am so sorry.",
    ),
    "despair": (
        "That makes me feel hopeless for a second, and I hate it.",
    ),
    "helplessness": (
        "I feel powerless here, and I am saying it out loud.",
    ),
}

# Emotions that can drive an impulse; calm, slow ones never blurt.
IMPULSE_PRONE = frozenset({
    "anger", "ill temper", "hatred", "disgust", "fear", "anxiety",
    "surprise", "joy", "high spirits", "sadness", "suffering", "weeping",
    "grief", "despair", "helplessness", "blushing", "shyness", "pride",
    "sulkiness",
})


def impulse_line(dominant: str, rng: random.Random) -> str:
    """A spoken impulse line for the dominant emotion (never empty)."""
    pool = IMPULSE_RESPONSES.get(dominant) or (
        "Okay - that hit me before I could even think about it.",
        "Whoa, that came out of me before thinking finished.",
    )
    return pool[rng.randrange(len(pool))]


class ResponseComposer:
    """Builds candidate replies and lets the Q-agent's strategy pick one.

    Candidates: direct answer, emotion-tinted reply, curious follow-up, and
    thought-sharing.  Personality tunes verbosity and warmth; a detected
    non-English language adds a localized greeting.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.last_scores: Dict[str, float] = {}
        # What the mind SHOWS in chat — assigned by the Mind from its
        # ExpressionOptions (presentation, not cognition; see core/options.py).
        self.expression = None
        # anti-repetition: the TEMPLATE of the last learning confirmation, so
        # two consecutive teaching turns never answer with the same sentence
        # — even when only a number inside it changed ("learned 1 new thing"
        # vs "learned 3 new things" is the same line).
        self._last_learn_reply: Optional[str] = None
        # anti-repetition for trained sayings: recent spoken-saying templates
        # so a trained mind rotates through its vocabulary instead of wearing
        # one favorite line out.
        self._recent_saying_templates: List[str] = []

    @staticmethod
    def _template_key(text: str) -> str:
        """Collapse a sentence to its template: numbers become '#'."""
        return re.sub(r"\d+(?:\.\d+)?", "#", text.strip().lower())

    def compose(self, ctx: CognitiveContext, action: str = "direct") -> Response:
        candidates = self._candidates(ctx)
        self.last_scores = {k: v[1] for k, v in candidates.items()}
        chosen_key = action if action in candidates else (
            "direct" if "direct" in candidates else next(iter(candidates)))
        # never let exploration bury a confident answer: when the mind KNOWS
        # (confident reasoning answer) or just used a tool successfully (the
        # calculator result IS the answer), non-embedding strategies —
        # including "trained" sayings and the curious follow-up — step aside
        # for the direct answer
        tool_answered = any(inv.success for inv in ctx.tool_invocations)
        confident = ((ctx.answer is not None and ctx.answer.confidence >= 0.65)
                     or tool_answered)
        if (chosen_key not in ("direct", "emotional", "share_thought")
                and confident and "direct" in candidates):
            chosen_key = "direct"
        # NEXT-TOKEN FLUENCY re-rank: when the language model is trained and
        # the chosen candidate is in a near-tie with another, prefer the one
        # that reads more like the language the mind has absorbed. A small,
        # bounded nudge — it breaks close ties without ever overpowering
        # knowledge or the learned style preference.
        chosen_key = self._fluency_rerank(ctx, candidates, chosen_key)
        text, base_conf = candidates[chosen_key]
        # CONVERSATIONAL RATE — a talkative mind carries the topic FURTHER:
        # one extra conversational sentence at rate 1.5, two at 2.0+, drawn
        # from related knowledge, other memories and her own sayings. Never
        # applied to tool answers (the calculator result IS the answer).
        if not tool_answered:
            text = self._extend_conversationally(ctx, text)
        text = self._with_language_greeting(ctx, text)
        # NOTE: goals are deliberately NOT spoken. They act as private
        # motivation (style bias chosen by the mind) — motivation is felt,
        # never recited.

        resp = Response(
            text=text,
            confidence=round(clamp(base_conf), 3),
            thoughts=[(t.stage.value, t.content, t.confidence) for t in ctx.thoughts],
            emotions=ctx.emotions.snapshot(),
            learned_facts=[str(f) for f in ctx.learned_facts],
            tool_results=[inv.summary() for inv in ctx.tool_invocations],
            capabilities=list(ctx.percepts.keys()),
            assets=list(ctx.assets),
            meta={"action": chosen_key, "candidate_scores": self.last_scores,
                  "cycle": ctx.cycle, "dominant_emotion": ctx.emotions.dominant(),
                  "valence": round(ctx.emotions.valence, 3)},
        )
        ctx.response = resp
        return resp

    # ------------------------------------------------------------------ candidates
    def _candidates(self, ctx: CognitiveContext) -> Dict[str, tuple]:
        cands: Dict[str, tuple] = {}

        direct = self._direct_text(ctx)
        if direct:
            cands["direct"] = (direct, self._direct_confidence(ctx))

        emotional = self._emotional_text(ctx, direct)
        if emotional:
            cands["emotional"] = (emotional, self._direct_confidence(ctx) * 0.95)

        curious = self._curious_text(ctx)
        if curious:
            cands["curious"] = (curious, 0.55)

        shared = self._share_thought_text(ctx)
        if shared:
            cands["share_thought"] = (shared, self._direct_confidence(ctx) * 0.9)

        # Trained sayings: the mind's own spoken vocabulary. Each trained line
        # is one more sentence shape available, so a well-trained mind has
        # more to draw from here — the structural fix for "she keeps repeating
        # the same sentence". Only fresh sayings are offered.
        saying = self._trained_text(ctx)
        if saying:
            cands["trained"] = (saying, 0.55)

        if not cands:
            dominant = ctx.emotions.dominant()
            if dominant and INTERJECTIONS.get(dominant):
                cands["emotional"] = (
                    f"{INTERJECTIONS[dominant]} I'm here and listening.", 0.45)
            else:
                cands["direct"] = (
                    "I'm listening — say something and I'll think about it.", 0.3)
        return cands

    # percept names that should never be the headline of a reply
    _BACKGROUND_PERCEPTS = {"embeddings"}
    # priority order for headline-worthy percepts (multilingual last: it
    # usually just tells us the input was English, which is not news)
    _HEADLINE_ORDER = ("media_generation", "vision", "audio", "code", "multilingual")

    def _direct_text(self, ctx: CognitiveContext) -> str:
        if ctx.tool_invocations:
            parts = [inv.summary() for inv in ctx.tool_invocations if inv.success]
            failures = [inv.summary() for inv in ctx.tool_invocations if not inv.success]
            if not parts and ctx.answer is not None:
                # Every speculative tool failed — the mind's own answer still
                # stands; a failed guess must not bury it.
                return ctx.answer.text
            lead = "I used my tools: " if parts else "I tried my tools but: "
            body = "; ".join(parts or failures)
            extra = f" {ctx.answer.text}" if ctx.answer is not None and parts else ""
            return lead + body + "." + extra
        if ctx.answer is not None:
            text = ctx.answer.text
            if ctx.learned_facts:
                # The turn both answered AND taught — acknowledge the new
                # knowledge briefly instead of burying the answer.
                n = len(ctx.learned_facts)
                text += " " + self.rng.choice([
                    f"I also picked up {n} new fact(s) from this.",
                    f"That also taught me {n} new thing(s), by the way.",
                    f"Filed {n} new fact(s) away while we were at it.",
                ])
            return text
        if ctx.learned_facts:
            names = "; ".join(str(f) for f in ctx.learned_facts[:3])
            templates = [
                f"Got it — I've learned that {names}. Ask me anything about it!",
                f"Thank you for teaching me — {names} is part of my knowledge now.",
                f"That clicked into place — {names} just connected for me.",
                f"I'm building quite the picture now — {names} went onto the shelf.",
                f"New knowledge absorbed — {names} is in my knowledge base.",
                f"That lesson landed — {names} grew what I know just now.",
            ]
            last_tpl = (self._template_key(self._last_learn_reply)
                        if self._last_learn_reply else None)
            fresh = [t for t in templates
                     if self._template_key(t) != last_tpl]
            chosen = self.rng.choice(fresh) if fresh else templates[0]
            self._last_learn_reply = chosen
            return chosen
        for name in self._HEADLINE_ORDER:
            if name in ctx.percepts and name not in self._BACKGROUND_PERCEPTS:
                percept = ctx.percepts[name]
                if name == "multilingual" and getattr(percept, "language", "en") == "en":
                    continue          # "it was English" is not a headline
                return percept.description
        for pname, percept in ctx.percepts.items():
            if pname in self._BACKGROUND_PERCEPTS:
                continue
            if pname == "multilingual" and getattr(percept, "language", "en") == "en":
                continue
            return percept.description
        return ""

    def _direct_confidence(self, ctx: CognitiveContext) -> float:
        if ctx.answer is not None:
            return ctx.answer.confidence
        if ctx.learned_facts:
            return 0.75
        for name in self._HEADLINE_ORDER:
            if name in ctx.percepts:
                return ctx.percepts[name].confidence
        for pname, percept in ctx.percepts.items():
            if pname not in self._BACKGROUND_PERCEPTS:
                return percept.confidence
        return 0.4

    def _emotional_text(self, ctx: CognitiveContext, direct: str) -> str:
        if not direct:
            return ""
        dominant = ctx.emotions.dominant()
        if dominant is None:
            return ""
        interjection = INTERJECTIONS.get(dominant, "")
        if not interjection:
            return direct
        # Standalone answer words keep their capital in every style —
        # "Oh, interesting — Yes — bird can fly." not "— yes —".
        first_word = direct.split(" ", 1)[0] if direct else ""
        keep_case = first_word.strip().lower() in ("yes", "no")
        body = direct if keep_case else (
            direct[0].lower() + direct[1:] if direct[:1].isupper() else direct)
        # Feelings are FELT, not announced: by default the reply just sounds
        # like the mood (the interjection IS the feeling showing). The
        # mechanical introspection — "(my current read: ...)" — appears only
        # when the developer turns show_feeling_reads on; the raw state stays
        # readable any time via mind.feeling.current_status.
        if self.expression is not None and self.expression.show_feeling_reads:
            return f"{interjection} (my current read: {dominant}). {body}"
        return f"{interjection} {body}"

    # trailing words that never make a good topic ("...would i have?")
    _TOPIC_FILLERS = {
        "have", "has", "had", "was", "were", "are", "did", "will",
        "would", "can", "could", "should", "may", "might", "must",
        "the", "and", "but", "for", "with", "about", "get", "got",
        # time/degree words never name a subject — "how many do i currently
        # have?" must not become "about currently"
        "currently", "now", "then", "today", "recently", "still", "already",
        "again", "always", "never", "maybe", "perhaps", "really", "actually",
        "just", "very", "many", "much", "also", "too", "quite", "pretty",
    }

    @classmethod
    def _memory_topic(cls, text: str) -> str:
        """Last meaningful word of a memory — questions put the topic last.
        When the tail token is thin ("many", "plus"), the last substantial
        content word is preferred. Hex-ish tokens (media artifact ids like
        aa9546f4...) are never topics."""
        words = [w for w in tokenize(text)
                 if len(w) > 2 and w not in cls._TOPIC_FILLERS
                 and not any(c.isdigit() for c in w)]
        while len(words) > 1 and words[-1] in cls._TOPIC_FILLERS:
            words.pop()
        for word in reversed(words):
            if len(word) >= 4:
                return word
        return words[-1] if words else "that"

    def _curious_text(self, ctx: CognitiveContext) -> str:
        # Follow-up questions are presentation, not cognition: the developer
        # turns them off with with_expression(lambda e: e.follow_ups(False))
        # and the mind only answers. The new-word weave is NOT a follow-up —
        # it is her own natural reaction to learning a word — and stays
        # available even then.
        ask_followups = (self.expression.ask_follow_up_questions
                         if self.expression is not None else True)
        # When the user talks about a person whose gender the mind does not
        # know, the question refers to that person with they/them — the ONLY
        # place they/them is used: describing unknown people, never the mind
        # itself.
        if self._mentions_unknown_person(ctx.text):
            text = self.rng.choice([
                "Who are they to you? I do not know them yet - what are they like?",
                "What are they like? I have not met them, so all I know is what you tell me.",
                "Tell me more about them - who are they, exactly?",
            ]) if ask_followups else ""
        elif ctx.scratch.get("epistemic") == "1":
            # Epistemic emotions drive questions (information-gap theory):
            # surprise or interest opens a gap, and the natural human move
            # is to ASK about it — "what is it?", "tell me more".
            topic = self._memory_topic(ctx.text)
            text = self.rng.choice([
                f"What is {topic}, exactly? You have me curious now.",
                f"Oh? Tell me more about {topic} - I am genuinely interested.",
                f"Now I want to know more about {topic} - what is it?",
                f"Wait - what is the whole story with {topic}? Tell me everything.",
            ]) if ask_followups else ""
        elif ctx.recalled:
            rec, _ = ctx.recalled[0]
            # Memory content is the user's own words — reference it obliquely
            # (topic only) instead of quoting it back verbatim.
            topic = self._memory_topic(rec.content)
            text = (f"That rings a bell — it links up with something you told "
                    f"me about {topic} earlier. Could you tell me more?") if ask_followups else ""
        else:
            # only curious about genuinely headline-worthy perceptions
            headline = None
            for name in self._HEADLINE_ORDER:
                if name not in ctx.percepts or name in self._BACKGROUND_PERCEPTS:
                    continue
                p = ctx.percepts[name]
                if name == "multilingual" and getattr(p, "language", "en") == "en":
                    continue
                headline = p.description
                break
            if headline is None:
                text = ""
            else:
                text = (f"I perceive something new: {headline} What do you make of it?"
                        if ask_followups else "")
        # Growing vocabulary: weave a freshly picked-up word into the reply —
        # each new word spawns a brand-new sentence, so the mind's spoken
        # variety grows WITH the conversation. The word is consumed only if
        # this candidate wins (the mind marks it spoken afterwards).
        fresh_word = ctx.scratch.get("fresh_word")
        if fresh_word:
            weave = self.rng.choice([
                f" By the way - I just picked up the word \"{fresh_word}\" from you. What does it mean in your world?",
                f" Also - \"{fresh_word}\" is a brand-new word for me. How do you use it?",
                f" And my vocabulary just grew by one: \"{fresh_word}\". What does it mean to you?",
            ])
            ctx.scratch["weave_word"] = fresh_word
            text = (text + weave) if text else weave.strip()
        return text

    # Words that introduce a PERSON whose gender the mind cannot know. When
    # one appears, the mind refers to that person with they/them — the only
    # context where those pronouns are used (never for the mind itself).
    _UNKNOWN_PERSON_WORDS = {
        "someone", "somebody", "person", "stranger", "friend", "buddy", "pal",
        "neighbor", "neighbour", "visitor", "guest", "colleague", "coworker",
        "classmate", "cousin", "sibling", "they", "them", "their",
    }

    @classmethod
    def _mentions_unknown_person(cls, text: str) -> bool:
        return any(t in cls._UNKNOWN_PERSON_WORDS for t in tokenize(text or ""))

    def _trained_text(self, ctx: CognitiveContext) -> str:
        """A trained saying to speak in the mind's own voice, or ''.
        Wordings the self-improvement ledger demoted are steered away from;
        approved ones are preferred."""
        sayings = ctx.scratch.get("sayings") or []
        improvement = ctx.scratch.get("improvement")

        def reputation(s: str) -> float:
            if improvement is None:
                return 0.0
            key = self._template_key(s)
            return improvement.template_boost(key) - 1.5 * improvement.template_penalty(key)

        fresh = [s for s in sayings
                 if self._template_key(s) not in self._recent_saying_templates]
        if not fresh:
            return ""
        fresh.sort(key=reputation, reverse=True)
        chosen = fresh[0]
        self._recent_saying_templates.append(self._template_key(chosen))
        if len(self._recent_saying_templates) > 3:
            self._recent_saying_templates.pop(0)
        return chosen

    def _share_thought_text(self, ctx: CognitiveContext) -> str:
        if not ctx.thoughts.to_list():
            return ""
        conclusion = ctx.thoughts.conclusion
        parts: List[str] = ["Here's my thinking:"]
        summary = ctx.percept_summary()
        if summary:
            parts.append(f"I perceived: {summary[:80]}")
        if ctx.derived_facts:
            parts.append(f"I derived {len(ctx.derived_facts)} new fact(s) by reasoning.")
        if (conclusion is not None
                and "nothing substantial" not in conclusion.content):
            parts.append(f"So: {conclusion.content}")
        if len(parts) == 1:
            return ""
        return " ".join(parts)

    # ------------------------------------------------------------------
    # CONVERSATIONAL RATE — how talkative she is (mind.conversational.rate).
    # Extras are declarative statements only (follow-up questions stay under
    # the expression.ask_follow_up_questions flag) and drawn from what the
    # cycle actually gathered: related knowledge, second memories, her own
    # sayings — with a warm fallback line so the rate always has effect.
    # ------------------------------------------------------------------
    def _extend_conversationally(self, ctx: CognitiveContext, text: str) -> str:
        conv = getattr(self, "conversational", None)
        if conv is None or conv.rate < 1.5:
            return text
        extras = 2 if conv.rate >= 2.0 else 1
        topic = self._memory_topic(ctx.text)
        if not topic or topic == "that":
            return text

        available: List[str] = []
        knowledge = getattr(self, "knowledge", None)
        if knowledge is not None:
            for fact in knowledge.facts():
                rendered = str(fact)
                subj = (getattr(fact, "subject", "") or "").lower()
                obj = (getattr(fact, "object", "") or "").lower()
                if len(rendered) > 8 and (topic in subj or topic in obj):
                    lowered = rendered[0].lower() + rendered[1:].rstrip(".")
                    available.append(
                        f"Come to think of it - {lowered}. So this topic "
                        "goes deeper than one line.")
                    break
        if len(ctx.recalled) > 1:
            side_topic = self._memory_topic(ctx.recalled[1][0].content)
            if side_topic and side_topic not in (topic, "that"):
                available.append(
                    f"This also brushes against what I recall about "
                    f"{side_topic} - one more thread I would love to pull "
                    "someday.")
        for saying in (ctx.scratch.get("sayings") or [])[:3]:
            if self._template_key(saying) != self._template_key(text):
                available.append(f'Somewhere in my own words, this fits: "{saying}"')
                break
        available.append(f"I could honestly keep talking about {topic} for a while.")
        available.append(f"Conversations that land on {topic} are the kind I remember.")

        added: List[str] = []
        text_key = self._template_key(text)
        for extra in available:
            if len(added) >= extras:
                break
            key = self._template_key(extra)
            if key == text_key or any(self._template_key(a) == key for a in added):
                continue
            added.append(extra)
        return text + " " + " ".join(added) if added else text

    # ------------------------------------------------------------------ styling
    # ------------------------------------------------------------------ fluency
    _FLUENCY_TIE_WINDOW = 0.15      # only break NEAR ties
    _FLUENCY_ADVANTAGE = 0.25       # and only when the other reads much better

    def _fluency(self, ctx: CognitiveContext, text: str) -> float:
        """0..1 fluency of a candidate under the mind's language model."""
        lm = getattr(ctx, "language_model", None)
        if lm is None or not lm.is_trained or not text:
            return 0.0
        import math
        mean_log = lm.score(text)
        return max(0.0, min(1.0, (mean_log + 5.0) / 5.0))

    def _fluency_rerank(self, ctx: CognitiveContext, candidates: dict,
                        chosen_key: str) -> str:
        if chosen_key not in ("trained", "curious", "emotional", "share_thought"):
            return chosen_key
        lm = getattr(ctx, "language_model", None)
        if lm is None or not lm.is_trained:
            return chosen_key
        chosen_conf = candidates[chosen_key][1]
        chosen_fluency = self._fluency(ctx, candidates[chosen_key][0])
        best_key, best_gain = chosen_key, 0.0
        for key, (text, conf) in candidates.items():
            if key == chosen_key or key not in ("trained", "curious", "emotional", "share_thought"):
                continue
            if abs(conf - chosen_conf) > self._FLUENCY_TIE_WINDOW:
                continue
            gain = self._fluency(ctx, text) - chosen_fluency
            if gain > best_gain:
                best_key, best_gain = key, gain
        if best_gain > self._FLUENCY_ADVANTAGE:
            return best_key
        return chosen_key

    def compose_impulse(self, ctx: CognitiveContext, dominant: str) -> Response:
        """The dual-process impulse response: an immediate first-person
        emotional reaction, spoken BEFORE thinking finishes. Used only when
        the mind has no confident answer (knowledge wins over blurting)."""
        text = impulse_line(dominant, self.rng)
        self.last_scores = {"impulse": 0.6}
        resp = Response(
            text=text,
            confidence=0.6,
            thoughts=[(t.stage.value, t.content, t.confidence) for t in ctx.thoughts],
            emotions=ctx.emotions.snapshot(),
            learned_facts=[str(f) for f in ctx.learned_facts],
            tool_results=[inv.summary() for inv in ctx.tool_invocations],
            capabilities=list(ctx.percepts.keys()),
            assets=list(ctx.assets),
            meta={"action": "impulse", "candidate_scores": self.last_scores,
                  "cycle": ctx.cycle, "dominant_emotion": dominant,
                  "valence": round(ctx.emotions.valence, 3),
                  "impulse": True},
        )
        ctx.response = resp
        return resp

    def _with_language_greeting(self, ctx: CognitiveContext, text: str) -> str:
        lang = ctx.scratch.get("language")
        greeting = ctx.scratch.get("localized_greeting")
        if lang and greeting and lang != "en" and text:
            return f"{greeting}! ({lang}) {text}"
        return text
