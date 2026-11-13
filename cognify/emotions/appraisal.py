"""OCC-flavored appraisal: turning stimuli into emotion deltas."""
from __future__ import annotations

from typing import Dict, Tuple

from ..util import clamp, tokenize
from .emotion_state import EMOTION_NAMES

PLEASANT = {
    "loved": .4, "great": .4, "happy": .45, "wonderful": .5, "good": .3,
    "thanks": .4, "thank": .35, "amazing": .5, "beautiful": .45, "nice": .25,
    "excellent": .45, "glad": .4, "awesome": .5, "like": .2, "enjoy": .3, "cool": .2,
    "perfect": .45, "brilliant": .4, "well": .15, "yes": .1, "success": .35,
}
UNPLEASANT = {
    "hate": .5, "awful": .5, "terrible": .5, "bad": .35, "sad": .45, "ugly": .35,
    "horrible": .5, "wrong": .25, "fail": .4, "failed": .45, "worst": .5, "boring": .3,
    "annoying": .35, "dislike": .35, "no": .08, "pain": .4, "hurt": .45, "loss": .4,
}
FEAR_WORDS = {
    "scared": .5, "afraid": .5, "fear": .5, "danger": .45, "dangerous": .45,
    "threat": .45, "worry": .35, "worried": .4, "anxious": .4, "panic": .5,
    "creepy": .35, "scary": .45, "frightening": .45, "dark": .12, "risk": .2,
}
ANGER_WORDS = {
    "angry": .5, "furious": .55, "mad": .4, "unfair": .4, "outrageous": .5,
    "stupid": .4, "idiot": .45, "betrayed": .5, "insult": .45, "hate you": .6,
}
SURPRISE_WORDS = {
    "wow": .4, "suddenly": .35, "unexpected": .45, "surprise": .5, "surprised": .5,
    "amazing": .2, "shock": .5, "whoa": .4, "incredible": .3, "never": .15,
}
TRUST_WORDS = {
    "trust": .4, "honest": .3, "reliable": .35, "friend": .3, "promise": .3,
    "believe": .25, "count on": .4, "loyal": .35,
}

# ---- Darwin's finer-grained feelings ---------------------------------------
# (The Expression of the Emotions in Man and Animals). Words already mapped
# above keep their original targets; these lexicons feed only the new
# emotions, so no earlier behavior silently changes.
LOVE_WORDS = {
    "love": .5, "loves": .45, "loved": .4, "loving": .4, "adore": .5,
    "adored": .45, "adorable": .4, "cherish": .45, "cherished": .4,
    "affection": .4, "darling": .35, "sweetheart": .35,
}
DEVOTION_WORDS = {
    "devotion": .5, "devoted": .45, "loyalty": .4, "dedication": .4,
    "dedicated": .35, "faithful": .35, "faithfully": .3,
}
TENDER_WORDS = {
    "tender": .4, "tenderness": .45, "gentle": .35, "gently": .3,
    "kindness": .4, "cozy": .3, "hug": .35, "hugs": .35, "hugged": .3,
    "cuddle": .35, "warmth": .3, "snuggle": .35,
}
SUFFERING_WORDS = {
    "suffering": .5, "suffer": .45, "suffers": .4, "suffered": .4,
    "agony": .55, "agonizing": .5, "torment": .5, "anguish": .5,
    "anguished": .45, "misery": .5, "excruciating": .55,
}
WEEPING_WORDS = {
    "weeping": .5, "weep": .45, "weeps": .4, "sob": .45, "sobs": .4,
    "sobbing": .5, "crying": .45, "wail": .4, "wailing": .45,
    "bawling": .45, "tearful": .35, "teary": .3,
}
HIGH_SPIRITS_WORDS = {
    "elated": .55, "elation": .5, "cheerful": .45, "cheerfulness": .45,
    "lively": .35, "buoyant": .4, "exhilarated": .55, "exhilarating": .45,
    "thrilled": .5, "thrilling": .4, "upbeat": .4, "glee": .45,
    "gleeful": .45, "jubilant": .5, "overjoyed": .5, "giddy": .4,
}
LOW_SPIRITS_WORDS = {
    "gloomy": .45, "gloom": .4, "downcast": .4, "melancholy": .45,
    "melancholic": .45, "morose": .4, "despondent": .45, "dispirited": .4,
    "downhearted": .4, "moping": .35, "mopey": .3, "heavyhearted": .4,
    "miserable": .5,
}
ANXIETY_WORDS = {
    "anxiety": .5, "uneasy": .45, "unease": .4, "apprehensive": .45,
    "apprehension": .4, "dread": .5, "dreaded": .45, "tense": .3,
    "restless": .35, "restlessness": .35, "jittery": .35,
}
GRIEF_WORDS = {
    "grief": .55, "grieving": .5, "grieve": .45, "grieved": .4,
    "mourning": .5, "mourn": .45, "mournful": .4, "heartbroken": .55,
    "heartbreak": .5, "bereaved": .45, "sorrow": .5, "sorrowful": .45,
}
DEJECTION_WORDS = {
    "dejected": .45, "dejection": .45, "deflated": .4, "discouraged": .4,
    "disheartened": .4, "crestfallen": .45, "demoralized": .4,
}
DESPAIR_WORDS = {
    "despair": .55, "despairing": .5, "hopeless": .5, "hopelessness": .55,
    "forlorn": .45,
}
HATRED_WORDS = {
    "hate": .5, "hates": .5, "hatred": .55, "hateful": .45, "loathe": .55,
    "loathing": .5, "detest": .55, "detestable": .5, "abhor": .55,
    "abhorrent": .5, "spite": .4, "spiteful": .4, "venomous": .4,
}
DISDAIN_WORDS = {
    "disdain": .5, "disdainful": .45, "scorn": .5, "scornful": .45,
    "deride": .4, "derision": .4, "derisive": .4,
}
CONTEMPT_WORDS = {
    "contempt": .5, "contemptuous": .45, "dismissive": .4, "sneer": .4,
    "sneering": .4, "belittling": .4,
}
GUILT_WORDS = {
    "guilt": .5, "guilty": .45, "ashamed": .5, "shame": .5, "shameful": .45,
    "remorse": .5, "remorseful": .45, "regretful": .4, "culpable": .35,
}
PRIDE_WORDS = {
    "pride": .45, "proud": .45, "proudly": .4, "triumphant": .5,
    "triumph": .45, "accomplished": .35,
}
HELPLESSNESS_WORDS = {
    "helpless": .5, "helplessness": .5, "powerless": .45,
    "powerlessness": .45, "trapped": .4, "futile": .35,
}
PATIENCE_WORDS = {
    "patience": .4, "patient": .3, "patiently": .35, "calmly": .3,
    "composure": .3, "composed": .25,
}
AFFIRMATION_WORDS = {
    "yes": .35, "yeah": .3, "yep": .3, "agreed": .35, "agrees": .3,
    "agreement": .25, "indeed": .3, "exactly": .35, "absolutely": .35,
    "definitely": .3, "affirm": .3, "affirmed": .3, "affirmation": .35,
    "confirmed": .25, "true": .2,
}
NEGATION_WORDS = {
    "no": .3, "nope": .35, "nah": .35, "disagree": .4, "disagrees": .35,
    "disagreement": .3, "wrong": .35, "false": .3, "incorrect": .35,
    "deny": .35, "denied": .35, "refute": .35, "refuted": .35,
    "negation": .35,
}
SELF_ATTENTION_WORDS = {
    "myself": .35, "spotlight": .35, "scrutinized": .4, "attention": .25,
    "watching": .3, "staring": .35,
}
SHYNESS_WORDS = {
    "shy": .45, "shyness": .45, "bashful": .4, "timid": .4, "timidity": .4,
    "reserved": .25, "reticent": .3, "wallflower": .35,
}
MODESTY_WORDS = {
    "modest": .35, "modesty": .4, "humble": .4, "humility": .4,
    "unassuming": .3, "unpretentious": .3,
}
BLUSHING_WORDS = {
    "blush": .45, "blushing": .5, "blushes": .4, "blushed": .4,
    "flushed": .4, "reddened": .35,
}
REFLECTION_WORDS = {
    "reflect": .35, "reflects": .3, "reflecting": .35, "reflected": .3,
    "reflection": .4, "introspect": .4, "introspection": .4,
    "introspective": .35, "ponder": .35, "pondering": .35, "musing": .3,
}
MEDITATION_WORDS = {
    "meditation": .4, "meditate": .4, "meditating": .4, "mediation": .3,
    "reverie": .35, "daydream": .35, "daydreaming": .35,
    "contemplation": .35, "contemplate": .35, "contemplating": .35,
    "contemplative": .35, "trance": .3,
}
ILL_TEMPER_WORDS = {
    "grumpy": .45, "grumpiness": .45, "cranky": .45, "irritable": .45,
    "irritability": .4, "grouchy": .45, "petulant": .4, "huffy": .35,
    "testy": .35, "snappy": .35,
}
SULKINESS_WORDS = {
    "sulking": .45, "sulk": .4, "sulky": .45, "sulkiness": .45,
    "pouting": .4, "pout": .35, "brooding": .35, "broody": .3,
    "moody": .35, "moodiness": .35, "sullen": .4, "sullenness": .4,
}
DETERMINATION_WORDS = {
    "determined": .45, "determination": .45, "resolve": .35, "resolute": .4,
    "resolutely": .35, "willpower": .4, "persist": .4, "persists": .35,
    "persistence": .4, "persistent": .4, "persevere": .4, "persevering": .4,
    "tenacity": .4, "tenacious": .4, "grit": .35, "gritty": .35,
    "undeterred": .4,
}

# (emotion name, lexicon, contribution factor) for every Darwin emotion.
_DARWIN_LEXICONS = (
    ("love", LOVE_WORDS, .8), ("devotion", DEVOTION_WORDS, .7),
    ("tenderness", TENDER_WORDS, .7), ("suffering", SUFFERING_WORDS, .7),
    ("weeping", WEEPING_WORDS, .7), ("high spirits", HIGH_SPIRITS_WORDS, .7),
    ("low spirits", LOW_SPIRITS_WORDS, .7), ("anxiety", ANXIETY_WORDS, .7),
    ("grief", GRIEF_WORDS, .7), ("dejection", DEJECTION_WORDS, .7),
    ("despair", DESPAIR_WORDS, .7), ("hatred", HATRED_WORDS, .7),
    ("disdain", DISDAIN_WORDS, .6), ("contempt", CONTEMPT_WORDS, .6),
    ("guilt", GUILT_WORDS, .7), ("pride", PRIDE_WORDS, .6),
    ("helplessness", HELPLESSNESS_WORDS, .7), ("patience", PATIENCE_WORDS, .6),
    ("affirmation", AFFIRMATION_WORDS, .7), ("negation", NEGATION_WORDS, .6),
    ("self attention", SELF_ATTENTION_WORDS, .6), ("shyness", SHYNESS_WORDS, .7),
    ("modesty", MODESTY_WORDS, .6), ("blushing", BLUSHING_WORDS, .7),
    ("reflection", REFLECTION_WORDS, .6), ("meditation", MEDITATION_WORDS, .6),
    ("ill temper", ILL_TEMPER_WORDS, .7), ("sulkiness", SULKINESS_WORDS, .7),
    ("determination", DETERMINATION_WORDS, .7),
)

#: Emotions amplified by neuroticism (negative) or extraversion (positive).
_NEGATIVE_SET = frozenset({
    "fear", "sadness", "disgust", "anger",
    "suffering", "weeping", "low spirits", "anxiety", "grief", "dejection",
    "despair", "hatred", "disdain", "contempt", "guilt", "helplessness",
    "ill temper", "sulkiness", "negation", "shyness", "blushing",
    "self attention",
})
_POSITIVE_SET = frozenset({
    "joy", "trust",
    "love", "devotion", "tenderness", "high spirits", "pride",
    "determination", "patience", "affirmation", "modesty",
})


def _count_hits(tokens, lexicon):
    hits = 0.0
    joined = " " + " ".join(tokens) + " "
    for phrase, weight in lexicon.items():
        if " " in phrase:
            if phrase in joined:
                hits += weight
        elif phrase in tokens:
            hits += weight
    return hits


# ---- Social appraisal triggers (appraisal research) -------------------------
# OCC: ANGER arises from blame (other-accountability) and unfairness — not
# from negativity alone — so insults aimed at ME and "that's unfair" weigh far
# more than ordinary bad words. Surprise markers set up the startle /
# information-gap response; interest markers carry the user's curiosity,
# which is socially contagious.
INSULT_WORDS = {
    "stupid", "dumb", "idiot", "idiotic", "useless", "worthless",
    "pathetic", "trash", "garbage", "annoying", "boring", "broken",
    "terrible", "awful", "horrible", "fool", "foolish", "lame",
    "loser", "failure", "sucks", "clueless",
}
SELF_INSULT_PHRASES = (
    "i hate you", "hate you", "you suck", "nobody likes you",
    "no one likes you", "nobody cares about you",
)
UNFAIR_WORDS = {"unfair", "rigged", "unjust", "biased"}
DISMISSAL_PHRASES = (
    "shut up", "be quiet", "stop talking", "i don't care", "i dont care",
    "who cares", "nonsense",
)
SURPRISE_PHRASES = (
    "guess what", "did you know", "believe it or not", "you won't believe",
    "you wont believe", "no way", "oh my god", "omg", "wait what",
    "you will not believe",
)
INTEREST_PHRASES = (
    "tell me more", "i wonder", "want to know", "wanna know",
    "how does that work", "why is that", "what is that", "what's that",
)
INTEREST_WORDS = {"interesting", "fascinating", "intriguing", "curious", "fascinated"}


def detect_social_triggers(text: str) -> Dict[str, bool]:
    """Detects the social-appraisal triggers of one utterance.

    Returns a dict with the keys ``self_insult``, ``unfairness``,
    ``dismissal``, ``surprise_marker`` and ``interest_marker``.
    Pure function, never raises.
    """
    out = {"self_insult": False, "unfairness": False, "dismissal": False,
           "surprise_marker": False, "interest_marker": False}
    if not text:
        return out
    lower = text.lower()
    tokens = tokenize(lower)

    # Blame aimed at ME: "you (are / re) <insult>" within a short window.
    for i, tok in enumerate(tokens):
        if tok == "you" and any(w in INSULT_WORDS for w in tokens[i + 1:i + 4]):
            out["self_insult"] = True
            break
    if not out["self_insult"]:
        out["self_insult"] = any(p in lower for p in SELF_INSULT_PHRASES)

    out["unfairness"] = any(w in tokens for w in UNFAIR_WORDS) or "not fair" in lower
    out["dismissal"] = any(p in lower for p in DISMISSAL_PHRASES)
    out["surprise_marker"] = any(p in lower for p in SURPRISE_PHRASES)
    out["interest_marker"] = (any(p in lower for p in INTEREST_PHRASES)
                              or any(w in tokens for w in INTEREST_WORDS))
    return out


def appraise_with_triggers(text: str, novelty: float = 0.0, goal_relevance: float = 0.0,
                           personality=None) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Like :func:`appraise`, but also returns the detected social triggers.

    The mind (and tests) use the trigger dict for the dual-process impulse
    gate: blame and dismissal are reflexive and answer before thinking.
    """
    tokens = tokenize(text)
    deltas: Dict[str, float] = {name: 0.0 for name in EMOTION_NAMES}

    pos = _count_hits(tokens, PLEASANT)
    neg = _count_hits(tokens, UNPLEASANT)
    deltas["joy"] += clamp(pos * 0.7)
    deltas["trust"] += clamp(_count_hits(tokens, TRUST_WORDS) * 0.6)
    deltas["sadness"] += clamp(neg * 0.55)
    deltas["disgust"] += clamp(neg * 0.35)
    deltas["fear"] += clamp(_count_hits(tokens, FEAR_WORDS) * 0.65)
    deltas["anger"] += clamp(_count_hits(tokens, ANGER_WORDS) * 0.6)
    deltas["surprise"] += clamp(_count_hits(tokens, SURPRISE_WORDS) * 0.5 + novelty * 0.35)
    deltas["anticipation"] += clamp(goal_relevance * 0.5 + (0.15 if tokens and tokens[-1] in ("will", "soon", "later", "tomorrow") else 0.0))

    # Darwin's finer feelings: each new emotion has its own vocabulary.
    for name, lexicon, factor in _DARWIN_LEXICONS:
        deltas[name] = clamp(deltas.get(name, 0.0) + clamp(_count_hits(tokens, lexicon) * factor))

    # Social appraisal triggers: blame and unfairness are what make someone
    # ANGRY (OCC); self-referential blame lands hardest. Interest is
    # socially contagious and feeds anticipation.
    triggers = detect_social_triggers(text)
    if triggers["self_insult"]:
        deltas["anger"] += 0.62
        deltas["sadness"] += 0.12          # anger blends with hurt
    if triggers["unfairness"]:
        deltas["anger"] += 0.35
    if triggers["dismissal"]:
        deltas["ill temper"] += 0.30
        deltas["sadness"] += 0.15
    if triggers["surprise_marker"]:
        deltas["surprise"] += 0.42
    if triggers["interest_marker"]:
        deltas["anticipation"] += 0.30

    # arousal amplifiers: exclamations / shouting
    if text:
        bangs = text.count("!")
        shouting = sum(1 for w in text.split() if len(w) > 2 and w.isupper())
        if bangs or shouting:
            amp = 1.0 + 0.12 * min(bangs, 3) + 0.15 * min(shouting, 3)
            for k in deltas:
                deltas[k] *= amp

    if personality is not None:
        amp_neg = 1.0 + 0.6 * personality.neuroticism
        amp_pos = 1.0 + 0.4 * personality.extraversion
        for k in deltas:
            if k in _NEGATIVE_SET:
                deltas[k] *= amp_neg
            elif k in _POSITIVE_SET:
                deltas[k] *= amp_pos
            if k == "surprise":
                deltas[k] *= 1.0 + 0.5 * personality.openness
            if k == "anger" and personality.agreeableness > 0.6:
                deltas[k] *= 0.6

    return ({k: clamp(v, 0.0, 1.0) for k, v in deltas.items()}, triggers)


def appraise(text: str, novelty: float = 0.0, goal_relevance: float = 0.0,
             personality=None) -> Dict[str, float]:
    """Map stimulus text to per-emotion deltas.

    ``novelty``: 0..1 fraction of unfamiliar tokens (drives surprise).
    ``goal_relevance``: 0..1 overlap with the active goal (drives anticipation).
    ``personality``: optional OCEAN profile modulating sensitivity.
    """
    return appraise_with_triggers(text, novelty=novelty, goal_relevance=goal_relevance,
                                  personality=personality)[0]
