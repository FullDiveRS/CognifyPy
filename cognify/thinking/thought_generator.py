"""The thought generator — the mind's silent self-talk BEFORE it speaks.

When thinking is enabled (``ThinkingOptions.enabled``), each cycle produces a
small set of internal thoughts — sized by the :class:`ThinkingLevel`
(Low=1 ... Max=8, or an explicit ``thought_budget``) and flavored by the
configured thinking tone. The thoughts are grounded in what the cycle
actually gathered — the topic, the current feeling, recalled memories,
freshly derived facts, the private goal and new vocabulary — so the final
reply ARRIVES from thought instead of appearing out of nowhere. Everything
recorded here lands on the chain of thought and ``mind.last_thoughts``;
whether the host SHOWS it is controlled by
``ExpressionOptions.show_thinking``.

Mirrors the C# ``Cognify.Thinking.ThoughtGenerator``.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from ..core.context import CognitiveContext
from ..core.options import ThinkingOptions
from ..compose.composer import ResponseComposer

__all__ = ["generate"]

# Generic deliberation used to keep the level's promise when this cycle
# offered fewer grounded observations than the level asks for (a bare
# "hello" at Max is only three grounded observations).
_FILLER_THOUGHTS: Tuple[str, ...] = (
    "Second look - is there anything in this I am missing?",
    "Weighing how much of this is fact and how much is just feeling.",
    "Keep the words simple and honest - say what I actually know.",
    "No need to rush the answer - one clear sentence beats three vague ones.",
    "Check the tone once more before speaking - match theirs, stay warm.",
    "Is there one small useful thing I can offer here? Lead with that.",
    "If I am unsure, say so plainly - honesty reads better than bluffing.",
    "What would make this reply worth their time? Answer, then stop.",
)


def _truncate(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def generate(ctx: CognitiveContext, options: ThinkingOptions,
             rng: random.Random) -> List[Tuple[str, float]]:
    """Generates the pre-speech thoughts: between 1 and 10 lines, never
    empty (the mind always thinks at least once before speaking when
    thinking is enabled). Returns ``(thought, confidence)`` pairs."""
    rounds = max(1, min(10, options.thought_rounds))
    tone = (options.tone or "balanced").strip().lower()
    topic = ResponseComposer._memory_topic(ctx.text)
    dominant = ctx.emotions.dominant()
    dominant_name = (str(dominant).lower()
                     if dominant and ctx.emotions.get(dominant) >= 0.05
                     else "steady")

    # The candidate pool: grounded observations of THIS cycle. Shuffled so
    # different levels think different things first, not just "more of the
    # same".
    pool: List[str] = []

    # 1. Knowledge check — do I know this?
    if ctx.answer is not None:
        pool.append(rng.choice([
            f"I know this one - {_truncate(ctx.answer.text)}. Lead with it.",
            f"Solid knowledge here: {_truncate(ctx.answer.text)}. Say it plainly.",
        ]))
    else:
        pool.append(rng.choice([
            f"What do I actually know about {topic}? Not enough - say so honestly and offer to learn.",
            f"No solid knowledge about {topic} yet. Do not fake it - invite them to teach me.",
        ]))

    # 2. Reasoning echo — what did inference just produce?
    if ctx.derived_facts:
        pool.append(rng.choice([
            f"My reasoning connected {len(ctx.derived_facts)} new thing(s) - "
            f"{_truncate(str(ctx.derived_facts[0]))}. Worth sharing.",
            f"Inference ran: {_truncate(str(ctx.derived_facts[0]))} - that can lead the reply.",
        ]))

    # 3. Feeling check — how should the reply FEEL?
    pool.append(rng.choice([
        f"This lands as {dominant_name} - let the feeling color the words, not dominate them.",
        f"I feel {dominant_name} about this. The reply should sound like that, without announcing it.",
    ]))

    # 4. Memory check — does anything echo?
    if ctx.recalled:
        memory_topic = ResponseComposer._memory_topic(ctx.recalled[0][0].content)
        pool.append(rng.choice([
            f"This echoes what they told me about {memory_topic} - nod to it obliquely, never quote it back.",
            "An old thread connects here - mention it lightly, one clause at most.",
        ]))

    # 5. Shape of the reply.
    pool.append({
        "focused": "Stay on the question. Answer, one follow-up at most, stop.",
        "playful": "Keep it light - a little bounce in the wording is fine today.",
        "poetic": "Give the reply one soft image - like light through a window - and no more.",
        "plain": "Short sentences. No decoration. Just the answer.",
    }.get(tone, "Answer first, stay warm, ask at most one question - then stop."))

    # 6. New vocabulary — maybe weave the fresh word in.
    fresh_word = ctx.scratch.get("fresh_word")
    if fresh_word:
        pool.append(f"\"{fresh_word}\" is still unplaced in my mind - "
                    "if it fits, ask about it in my own words.")

    # 7. The private goal — felt, never spoken about.
    if ctx.active_goal is not None:
        pool.append(f"Quiet motive in view ({ctx.active_goal.description}) - "
                    "let it warm the tone without ever naming it.")

    # 8. Epistemic gap — curiosity wants to ask.
    if ctx.scratch.get("epistemic") == "1":
        pool.append("A gap just opened - I want to know more. "
                    "One curious question is natural, two is pestering.")

    # Shuffle, then take the level's share (pad with generic deliberation so
    # the level's "token set" promise is always kept).
    rng.shuffle(pool)
    taken = pool[:rounds]
    filler = 0
    while len(taken) < rounds:
        taken.append(_FILLER_THOUGHTS[filler % len(_FILLER_THOUGHTS)])
        filler += 1

    # Tone garnish: a light lead-in word that flavors HOW it is said without
    # changing what is thought.
    if tone == "playful":
        taken = [f"Ooh - {t[0].lower() + t[1:]}" if rng.random() < 0.5 else t
                 for t in taken]
    elif tone == "focused":
        taken = [f"Focus: {t[0].lower() + t[1:]}" for t in taken]

    return [(t, round(rng.uniform(0.45, 0.70), 3)) for t in taken]
