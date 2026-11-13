"""ChainOfThought: the seven-stage deliberate thinking process."""
from __future__ import annotations

from typing import List, Optional

from ..core.context import CognitiveContext
from ..core.thought_stream import ThoughtStage, ThoughtStream
from ..util import tokenize


class ChainOfThought:
    """Produces a structured 7-stage thought stream for one cognitive cycle.

    Stages: perceive -> question -> hypothesize -> evidence -> evaluate ->
    conclude -> reflect.  Every stage is grounded in the cycle's context:
    percepts, memory recall, derived facts, tools, and emotional state.
    """

    def run(self, ctx: CognitiveContext) -> ThoughtStream:
        stream = ThoughtStream()
        self._stage_perceive(ctx, stream)
        self._stage_question(ctx, stream)
        self._stage_hypothesize(ctx, stream)
        self._stage_evidence(ctx, stream)
        self._stage_evaluate(ctx, stream)
        self._stage_conclude(ctx, stream)
        self._stage_reflect(ctx, stream)
        return stream

    # ------------------------------------------------------------------ stages
    def _stage_perceive(self, ctx: CognitiveContext, s: ThoughtStream) -> None:
        parts: List[str] = []
        if ctx.stimulus.text:
            parts.append(f'heard "{ctx.stimulus.text[:60]}"')
        for percept in ctx.percepts.values():
            parts.append(percept.description or f"{percept.modality} input")
        if not parts:
            parts.append("sensed silence")
        mood = ctx.emotions.dominant()
        suffix = f" (feeling {mood})" if mood else ""
        s.add(ThoughtStage.PERCEIVE, "; ".join(parts) + suffix, 0.85)

    def _stage_question(self, ctx: CognitiveContext, s: ThoughtStream) -> None:
        if ctx.has_question:
            s.add(ThoughtStage.QUESTION, "The user is asking something — I should answer it.",
                  0.9)
        elif ctx.active_goal is not None:
            s.add(ThoughtStage.QUESTION,
                  f"How does this relate to my goal: {ctx.active_goal.description}?", 0.6)
        else:
            s.add(ThoughtStage.QUESTION, "Is there anything new to learn here?", 0.5)

    def _stage_hypothesize(self, ctx: CognitiveContext, s: ThoughtStream) -> None:
        hyps: List[str] = []
        if ctx.recalled:
            rec, act = ctx.recalled[0]
            hyps.append(f"this relates to what I remember: '{rec.content[:50]}'")
        if ctx.percepts:
            hyps.append("the non-text input carries meaning I should combine with the words")
        if not hyps:
            hyps.append("this may be new information worth storing")
        for h in hyps[:2]:
            s.add(ThoughtStage.HYPOTHESIZE, h, 0.55)

    def _stage_evidence(self, ctx: CognitiveContext, s: ThoughtStream) -> None:
        bits: List[str] = []
        if ctx.knowledge_size:
            bits.append(f"{ctx.knowledge_size} facts known")
        if ctx.derived_facts:
            bits.append(f"{len(ctx.derived_facts)} newly derived")
        if ctx.recalled:
            bits.append(f"{len(ctx.recalled)} memories activated")
        if ctx.tool_invocations:
            for inv in ctx.tool_invocations:
                bits.append(f"tool {inv.name} -> {'ok' if inv.success else 'error'}")
        if ctx.answer is not None and ctx.answer.chain:
            bits.append("derivation chain available")
        if not bits:
            bits.append("no prior evidence found")
        conf = 0.8 if ctx.answer is not None else 0.5
        s.add(ThoughtStage.EVIDENCE, "; ".join(bits), conf)

    def _stage_evaluate(self, ctx: CognitiveContext, s: ThoughtStream) -> None:
        if ctx.answer is not None:
            q = "high" if ctx.answer.confidence > 0.7 else "moderate"
            s.add(ThoughtStage.EVALUATE,
                  f"Answer quality looks {q} (confidence {ctx.answer.confidence:.2f}).",
                  ctx.answer.confidence)
        elif ctx.learned_facts:
            s.add(ThoughtStage.EVALUATE,
                  f"No question to answer, but {len(ctx.learned_facts)} fact(s) "
                  f"were extracted and stored.", 0.7)
        else:
            mood = ctx.emotions.dominant()
            note = f" My {mood} colors my reading of this." if mood else ""
            s.add(ThoughtStage.EVALUATE, "Little evidence to weigh here." + note, 0.4)

    def _stage_conclude(self, ctx: CognitiveContext, s: ThoughtStream) -> None:
        if ctx.answer is not None:
            s.add(ThoughtStage.CONCLUDE, ctx.answer.text, ctx.answer.confidence)
        elif ctx.learned_facts:
            names = ", ".join(str(f) for f in ctx.learned_facts[:2])
            s.add(ThoughtStage.CONCLUDE,
                  f"Learned: {names}. Ready for questions about it.",
                  0.75)
        else:
            for name, percept in ctx.percepts.items():
                if name == "embeddings":
                    continue
                if getattr(percept, "language", None) == "en":
                    continue          # background percepts are not conclusions
                s.add(ThoughtStage.CONCLUDE, percept.description, percept.confidence)
                return
            s.add(ThoughtStage.CONCLUDE,
                  "I heard you, but I have nothing substantial to add yet.", 0.35)

    def _stage_reflect(self, ctx: CognitiveContext, s: ThoughtStream) -> None:
        bits: List[str] = []
        if ctx.learned_facts or ctx.derived_facts:
            bits.append(f"knowledge grew by {len(ctx.learned_facts) + len(ctx.derived_facts)}")
        valence = ctx.emotions.valence
        if valence > 0.15:
            bits.append("this felt good")
        elif valence < -0.15:
            bits.append("this felt uncomfortable")
        if ctx.active_goal is not None and ctx.active_goal.progress > 0.02:
            bits.append(f"still working on '{ctx.active_goal.description}'")
        text = "; ".join(bits) if bits else "nothing new to consolidate"
        s.add(ThoughtStage.REFLECT, text, 0.6)
