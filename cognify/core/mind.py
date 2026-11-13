"""Mind: the cognitive kernel that runs the full loop every cycle.

Loop order per cycle:
    perception -> memory -> emotions -> reasoning -> goals -> thinking ->
    tools -> learning -> composition -> response
"""
from __future__ import annotations

import urllib.parse
from typing import Any, Dict, List, Optional

from ..compose.composer import ResponseComposer, IMPULSE_PRONE
from ..emotions.appraisal import appraise_with_triggers
from ..emotions.emotion_state import EmotionalState
from ..emotions.feeling import FeelingSnapshot
from ..emotions.personality import Personality
from ..goals.goals import GoalManager
from ..learning.encoder import HashingEncoder
from ..learning.hybrid_next_token import fluency as _hybrid_fluency
from ..learning.hybrid_next_token import predict_next as _hybrid_predict
from ..learning.learner import LearningModule
from ..learning.next_token import NextTokenModel, TokenPrediction
from ..learning.self_improvement import SelfImprovement, SelfImprovementReport
from ..learning.transformer import TransformerLM, TransformerTrainingReport
from ..memory.memory import AssociativeMemory
from ..reasoning.facts import Fact, FactParser
from ..reasoning.inference import InferenceEngine
from ..reasoning.knowledge import KnowledgeBase
from ..reasoning.arithmetic import ArithmeticEngine
from ..thinking.chain_of_thought import ChainOfThought
from ..thinking.thought_generator import generate as _generate_thoughts
from ..util import clamp, tokenize
from .context import CognitiveContext, Response, ReasoningAnswer
from .events import EventBus
from .gender import GenderOptions, UNSPECIFIED
from .options import (ConversationalOptions, ExpressionOptions,
                      ThinkingCeiling, ThinkingLevel, ThinkingOptions)
from .phrases import PhrasePool
from .stimulus import Modality, Stimulus
from .thought_stream import ThoughtStage

# Lines that assert nothing (questions, conditionals) never become sayings —
# they are requests or setups, not things to say.
_QUESTION_STARTERS = frozenset({
    "what", "who", "when", "where", "why", "how", "is", "are", "do", "does",
    "did", "can", "could", "would", "will", "tell", "say", "name"})


def _is_sayable(line: str) -> bool:
    t = line.strip()
    if len(t) < 3 or len(t) > 300 or "?" in t:
        return False
    first = t.split()[0].strip(",.!:;").lower() if t.split() else ""
    return first != "if" and first not in _QUESTION_STARTERS


class TrainingReport:
    """What one training pass added: lines read, facts learned, sayings added
    to the mind's spoken vocabulary and number lines taught to its arithmetic."""

    def __init__(self) -> None:
        self.lines = 0
        self.facts_learned = 0
        self.sayings_added = 0
        self.sequences_taught = 0

    def __str__(self) -> str:
        return (f"{self.lines} line(s): {self.facts_learned} fact(s), "
                f"{self.sayings_added} saying(s), {self.sequences_taught} number line(s).")

    def __repr__(self) -> str:  # pragma: no cover
        return f"TrainingReport({self!s})"


class Mind:
    """A cognitive agent: perceives, feels, reasons, thinks, learns, acts."""

    def __init__(self, *, name: str = "Cognify", seed: int = 42,
                 personality: Personality, emotion_sensitivity: float,
                 memory_capacity: int, encoder_dims: int,
                 gender: Optional[GenderOptions] = None,
                 thinking: Optional[ThinkingOptions] = None,
                 expression: Optional[ExpressionOptions] = None,
                 conversational: Optional[ConversationalOptions] = None,
                 allow_web_search: bool = False,
                 auto_fetch_web: bool = False,
                 web_search_backend: Any = None,
                 explore_max_pages: int = 3,
                 flags: Dict[str, bool], capabilities: List[Any],
                 tools_registry: Any, user_tools: List[Any]) -> None:
        self.name = name
        self.seed = seed
        self.events = EventBus()
        self.cycle_count = 0
        self.personality = personality
        self.gender = gender or GenderOptions()
        self.emotions = EmotionalState(sensitivity=emotion_sensitivity)
        self.flags = dict(flags)
        # Thinking levels + expression (what chat shows) — developer-facing
        # options, mirror of the C# ThinkingOptions / ExpressionOptions.
        self.thinking_options = thinking or ThinkingOptions()
        self.expression = expression or ExpressionOptions()
        # Conversational options — how TALKATIVE she is and how much she
        # collects from the conversation (mirror of C# ConversationalOptions).
        self.conversational = conversational or ConversationalOptions()
        # Web switches (manual-only by default; see search_web/explore_url).
        self.allow_web_search = bool(allow_web_search)
        self.auto_fetch_web = bool(auto_fetch_web)
        self.explore_max_pages = max(1, min(6, int(explore_max_pages)))
        # The thought generator's output from the LAST cycle — the silent
        # self-talk produced before the reply was composed. Empty when
        # thinking is disabled. Hosts show it only when
        # expression.show_thinking is on.
        self.last_thoughts: List[str] = []
        self.encoder = HashingEncoder(dims=encoder_dims)
        self.kb = KnowledgeBase()
        self.parser = FactParser()
        self.engine = InferenceEngine()
        self.arithmetic = ArithmeticEngine()
        self.cot = ChainOfThought()
        self.composer = ResponseComposer(seed=seed)
        self.composer.expression = self.expression
        self.composer.conversational = self.conversational
        # The composer's conversational extension reads related knowledge.
        self.composer.knowledge = self.kb
        self.phrases = PhrasePool()
        self.tools = tools_registry
        self._user_tools = list(user_tools)
        self.capabilities = list(capabilities)

        # memory (embedding-aware similarity when the embedder is on)
        self._embedder = next((c for c in self.capabilities
                               if getattr(c, "name", "") == "embeddings"), None)
        self.memory = AssociativeMemory(capacity=memory_capacity,
                                        similarity=self._similarity)

        # goals
        self.goals = GoalManager()
        self.goals.drives["curiosity"].growth += 0.02 * personality.openness

        # learning
        self.learner: Optional[LearningModule] = None
        if self.flags.get("learning", True):
            self.learner = LearningModule(self.encoder, seed=seed)
        self._last_reward: Optional[float] = None
        self._novel_seen: set = set()
        # topics the self-grown-sayings mechanism already covered
        self._grown_from: set = set()

        # Next-token prediction: an n-gram language model that observes
        # everything the mind reads and says (conversation, training corpora,
        # web results) and powers fluency-aware composition + /predict.
        self.language_model = NextTokenModel()
        # The NEURAL half: a tiny decoder-only transformer (pure Python,
        # no dependencies) that observes the same text stream and trains
        # lazily — automatic when enough language accumulates, or explicit
        # via train_deep_model(). Predictions blend both models.
        self.neural_language_model = TransformerLM()
        # Self-improvement: implied feedback from the user's next message
        # grades the previous reply (wording reputations + Q rewards).
        self.improvement = SelfImprovement()
        # Pluggable web-search backend (manual-only — see search_web()).
        self.web_search_backend = web_search_backend
        # Conversational understanding: the talk's last topic (topic-shift
        # notes live in memory with the "conversation" tag).
        self._last_conversation_topic: Optional[str] = None
        # Topics the autonomous lookup already tried this session — one
        # lookup per unfamiliar topic, success or fail.
        self._auto_lookup_topics: set = set()

    # ------------------------------------------------------------------ public
    def process(self, stimulus: Stimulus | str) -> Response:
        """Run one full cognitive cycle on a stimulus."""
        if isinstance(stimulus, str):
            stimulus = Stimulus.text(stimulus)
        self.cycle_count += 1
        cycle = self.cycle_count
        self.events.emit("stimulus_received", cycle, text=stimulus.text[:100],
                         modality=stimulus.modality.value)

        ctx = CognitiveContext(cycle=cycle, stimulus=stimulus,
                               personality=self.personality, emotions=self.emotions)

        # CONVERSATIONAL UNDERSTANDING — when with_conversational(understand)
        # is on (the default), the mind tracks how the talk moves from topic
        # to topic; each shift leaves a compact note in memory (tag
        # "conversation"), so recall knows the SHAPE of the conversation.
        self._track_conversation_topic(stimulus)

        # 1. perception (capabilities, phase="perceive")
        for cap in self._caps("perceive"):
            if not cap.enabled:
                continue
            try:
                if cap.can_handle(stimulus, ctx):
                    percept = cap.handle(stimulus, ctx)
                    if percept is not None:
                        ctx.percepts[cap.name] = percept
                        self.events.emit("capability_result", cycle,
                                         capability=cap.name, description=percept.description)
            except Exception as exc:
                self.events.emit("capability_error", cycle, capability=cap.name,
                                 error=str(exc))

        # 2. memory
        novelty = self._novelty(stimulus.text)
        ctx.scratch["novelty"] = novelty
        # Vocabulary growth: listen for words the mind has never heard
        # before. The composer may weave them back into a reply — every new
        # word is one more thing the mind can say.
        if self.learner is not None and stimulus.text:
            picked = self.learner.lexicon.observe(stimulus.text, cycle)
            if picked:
                ctx.scratch["new_words"] = picked
        if self.flags.get("memory", True):
            ctx.recalled = self.memory.recall(stimulus.text or "perception", k=3)
            if stimulus.text:
                self.memory.add(stimulus.text, kind="episode",
                                importance=clamp(stimulus.salience))
            ctx.working_memory = [r for r, _ in ctx.recalled]
            self.events.emit("memory_recalled", cycle,
                             count=len(ctx.recalled), novelty=round(novelty, 2))

        # 3. emotions
        if self.flags.get("emotions", True):
            goal_relevance = self._goal_relevance(stimulus.text)
            deltas, triggers = appraise_with_triggers(
                stimulus.text, novelty=novelty,
                goal_relevance=goal_relevance,
                personality=self.personality)
            self.emotions.apply(deltas)

            # Expectation violation (appraisal research: surprise IS
            # violated expectation): a statement that negates something the
            # mind knows startles it.
            contradicts_known = self._contradicts_known(stimulus.text)
            if contradicts_known:
                self.emotions.boost("surprise", 0.45)

            self.events.emit("emotion_changed", cycle,
                             dominant=self.emotions.dominant(),
                             valence=round(self.emotions.valence, 3))

            # Dual-process gate (Zajonc's affect primacy): usually the mind
            # thinks BEFORE it speaks — but a strong enough surge answers
            # from feeling alone. Blame and dismissal are reflexive; other
            # emotions must cross the surge threshold. The composition step
            # honors the flag only when there is no confident answer.
            surge = max(deltas.items(), key=lambda kv: kv[1]) if deltas else (None, 0.0)
            reflexive = (triggers.get("self_insult") or triggers.get("dismissal")
                         or contradicts_known)
            if (surge[0] in IMPULSE_PRONE
                    and (reflexive or surge[1] >= 0.55)):
                ctx.scratch["impulse"] = surge[0]
            else:
                ctx.scratch.pop("impulse", None)

            # Epistemic surge (information-gap theory): a FRESH surprise or
            # spike of interest opens a knowledge gap — the mind will want
            # to ASK this turn. Flagged from the fresh appraisal (not the
            # standing state) so one spike cannot latch question mode.
            # Never on question turns: a question already IS the gap being
            # closed, so answering outranks asking.
            if (not ctx.has_question
                    and (deltas.get("surprise", 0.0) > 0.25
                         or deltas.get("anticipation", 0.0) > 0.25)):
                ctx.scratch["epistemic"] = "1"
            else:
                ctx.scratch.pop("epistemic", None)

        # 4. reasoning
        if self.flags.get("reasoning", True):
            ctx.learned_facts = self.parser.parse(stimulus.text)
            for fact in ctx.learned_facts:
                if self.kb.add(fact):
                    self.events.emit("fact_learned", cycle, fact=str(fact))
            # A taught counting sequence ("0, 1, 2, ..., 9") becomes the
            # mind's number line — counting along it later is reasoning from
            # teaching, not calculation, so no calculator is ever needed for
            # sums that stay on the taught line.
            self.arithmetic.teach_sequence(stimulus.text)
            derived = self.engine.forward_chain(self.kb)
            derived += self.engine.induce(self.kb, min_instances=3)
            ctx.derived_facts = derived
            for fact in derived:
                self.events.emit("fact_derived", cycle, fact=str(fact))
            ctx.knowledge_size = self.kb.size()
            identity = self._identity_answer(stimulus.text)
            if identity is not None:
                ctx.answer = identity
            else:
                # Taught arithmetic first: "what is 2 + 2?" is answered from
                # facts the user taught (or derived from them) — the
                # calculator tool only fills in when this comes up empty.
                math_result = (self.arithmetic.try_answer_question(self.kb, stimulus.text)
                               or self.arithmetic.try_word_problem(self.kb, stimulus.text))
                if math_result is not None:
                    ctx.answer = ReasoningAnswer(text=math_result.reply,
                                                 confidence=math_result.confidence,
                                                 kind="arithmetic")
                    ctx.scratch["math_answered"] = True
                elif (self.arithmetic.looks_like_arithmetic_question(stimulus.text)
                      or self.arithmetic.looks_like_word_problem(stimulus.text)):
                    # recognized as arithmetic but unanswerable from taught
                    # knowledge — the tool phase may transparently help out
                    ctx.scratch["math_attempted"] = True
                elif stimulus.text and (ctx.has_question or "?" in stimulus.text):
                    ctx.answer = self.engine.answer(self.kb, stimulus.text)

        # 5. goals
        if self.flags.get("goals", True):
            self.goals.tick()
            self.goals.consider(stimulus.text)
            self.goals.report_learning(len(ctx.learned_facts) + len(ctx.derived_facts))
            ctx.active_goal = self.goals.select_active()
            if ctx.active_goal is not None:
                self.events.emit("goal_updated", cycle,
                                 goal=ctx.active_goal.description,
                                 progress=round(ctx.active_goal.progress, 2))
            ctx.scratch["goal_progress"] = self.goals.progress_text()

        # 6. deliberate thinking
        if self.flags.get("thinking", True):
            # A fresh word the composer may weave into the curious candidate
            # — and the thought generator may think about — looked up here so
            # both can see it (spoken at most once per word, see after compose).
            if self.learner is not None:
                ctx.scratch["fresh_word"] = self.learner.lexicon.fresh_word()

            # 6a. The seven-stage deliberate pass.
            ctx.thoughts = self.cot.run(ctx)
            for t in ctx.thoughts:
                self.events.emit("thought", cycle, stage=t.stage.value, text=t.content)

            # 6b. GENERATE — the thought generator produces the mind's silent
            # self-talk BEFORE anything is composed (thinking levels Low..Max
            # size it; the tone flavors it). This is the "generate its
            # thoughts before speaking" phase: the reply is the RESULT the
            # mind arrives at after these thoughts, not the first thing it
            # does. An impulse skips the generator by definition — it speaks
            # from feeling alone, before thinking finishes.
            # THINKING AUTOMATIC — when enabled, the mind picks its own
            # effort for THIS cycle from the actual input; the configured
            # level is only the base. The choice is clamped by the ceiling
            # and recorded in the chain of thought. An explicit thought
            # budget overrides the choice.
            effective_thinking = self.thinking_options
            if (self.thinking_options.enabled and self.thinking_options.automatic
                    and self.thinking_options.thought_budget <= 0
                    and not ctx.scratch.get("impulse")):
                chosen = self._choose_automatic_level(ctx)
                effective_thinking = self.thinking_options.for_level(chosen)
                note = (f"Automatic: chose to think at {chosen.value} for "
                        "this input.")
                ctx.thoughts.add(ThoughtStage.PERCEIVE, note, 0.6)
                self.events.emit("thought_generated", cycle, text=note)
            if self.thinking_options.enabled and not ctx.scratch.get("impulse"):
                generated = _generate_thoughts(ctx, effective_thinking,
                                               self.composer.rng)
                for thought, conf in generated:
                    ctx.thoughts.add(ThoughtStage.GENERATE, thought, conf)
                    self.events.emit("thought_generated", cycle, text=thought)
                self.last_thoughts = [t for t, _ in generated]
            else:
                self.last_thoughts = []

            # The impulse path records why deliberation was cut short, so
            # the chain of thought stays explainable.
            if ctx.scratch.get("impulse"):
                ctx.thoughts.add(
                    ThoughtStage.CONCLUDE,
                    f"A surge of {ctx.scratch['impulse']} — answered before "
                    "thinking finished (impulse).", 0.6)
        else:
            self.last_thoughts = []

        # 7. tools (phase="tools" capabilities)
        for cap in self._caps("tools"):
            if not cap.enabled:
                continue
            try:
                if cap.can_handle(stimulus, ctx):
                    cap.handle(stimulus, ctx)
                    for inv in ctx.tool_invocations:
                        self.events.emit("tool_invoked", cycle, tool=inv.name,
                                         success=inv.success)
                        self.goals.report_tool_success(inv.success)
            except Exception as exc:
                self.events.emit("capability_error", cycle, capability=cap.name,
                                 error=str(exc))

        # 7b. Word problems the mind could not reason out from taught
        #     knowledge ("if i have 2 apples and buy 3 more..."): the
        #     calculator fills in transparently, computing the extracted
        #     quantities mechanically. Reasoning still gets first crack —
        #     this only fires when the reasoning step flagged math_attempted.
        lowered = (stimulus.text or "").lower()
        explicit_calc = ("calculator" in lowered
                         or lowered.startswith("calculate")
                         or lowered.startswith("compute"))
        if (ctx.scratch.get("math_attempted")
                and not ctx.scratch.get("math_answered")
                and not explicit_calc
                and not any(inv.name == "calculator" for inv in ctx.tool_invocations)
                and any(getattr(c, "name", "") == "tool_use" and c.enabled
                        for c in self.capabilities)):
            mechanical = self.arithmetic.try_word_problem_mechanically(stimulus.text or "")
            if mechanical is not None:
                # Spoken through ctx.answer: clean text, and the composer
                # forces a confident direct answer onto the Q-agent's choice.
                ctx.answer = ReasoningAnswer(
                    text="I have not been taught that one yet, so I used the "
                         f"calculator: {mechanical.reply}",
                    confidence=0.65, kind="arithmetic")
                self.events.emit("tool_invoked", cycle, tool="calculator", success=True)

        # 7c. AUTONOMOUS WEB LOOKUP — her own WebFetch (with_auto_web_fetch).
        #     When the conversation landed on a topic clearly OUTSIDE her
        #     knowledge — nothing recalled, no taught answer — she says so
        #     naturally, looks it up herself, learns from the results and
        #     answers from what she found. Still guarded: arithmetic,
        #     identity/personal talk and smalltalk never trigger it, one
        #     lookup per unfamiliar topic, and a failed lookup stays silent
        #     (the normal honest "teach me" reply runs instead).
        if (self.auto_fetch_web and self.flags.get("tools", True)
                and ctx.answer is None and not ctx.recalled
                and not ctx.tool_invocations
                and not ctx.scratch.get("math_answered")
                and not ctx.scratch.get("math_attempted")):
            self._try_knowledge_gap_lookup(ctx)

        # 8. learning (net training + Q credit assignment)
        if self.learner is not None:
            self.learner.observe(ctx)

        # 9. memory consolidation tick
        if self.flags.get("memory", True):
            self.memory.tick()

        # 10. compose the response
        # Trained sayings relevant to this input are offered to the composer
        # (its "trained" candidate) — the mind's own spoken vocabulary.
        if self.phrases.count:
            ctx.scratch["sayings"] = self.phrases.pick_relevant(
                stimulus.text or "", self.composer.rng, 4)
        # Self-improvement reputations steer the trained-saying choice away
        # from wordings that historically drew disapproval.
        ctx.scratch["improvement"] = self.improvement
        # The composer reads the language model for fluency-aware re-ranking.
        ctx.language_model = self.language_model
        # A freshly picked-up word the composer may weave into the curious
        # candidate — spoken at most once per word (see after compose).
        # (fresh_word was already looked up at step 6 — the thought generator
        # and the composer share it.)
        action = "direct"
        if self.learner is not None:
            action = self.learner.choose_action(ctx)
        # Dual-process: the impulse answer speaks BEFORE deliberation would —
        # but only when the mind has nothing confident to say (knowledge
        # wins over blurting).
        impulse = ctx.scratch.get("impulse")
        if impulse and (ctx.answer is None or ctx.answer.confidence < 0.65):
            response = self.composer.compose_impulse(ctx, impulse)
            # Credit the impulse choice so learning can shape how often the
            # mind gives in to the surge.
            if self.learner is not None:
                self.learner.credit_action("impulse")
        else:
            # Goal internalization: the private goal biases WHICH STYLE feels
            # right — a prosocial goal makes warm styles more likely without
            # ever being spoken about.
            action = self._goal_style_bias(ctx, action)
            # Epistemic override (information-gap theory): when THIS turn's
            # appraisal surged with surprise or interest, the mind ASKS — a
            # knowledge gap pulls harder than any learned style preference.
            # Never on question turns: a question already IS the gap being
            # closed, so answering outranks asking.
            if (ctx.scratch.get("epistemic") == "1"
                    and not ctx.has_question
                    and not (ctx.answer is not None and ctx.answer.confidence >= 0.65)):
                action = "curious"
            response = self.composer.compose(ctx, action)
        # If the winning candidate actually spoke the woven word, consume it
        # so it is never asked about again.
        if (self.learner is not None and action == "curious"
                and ctx.scratch.get("weave_word")):
            self.learner.lexicon.mark_spoken(ctx.scratch["weave_word"])
        self.events.emit("response_composed", cycle, action=action,
                         text=response.text[:100], confidence=response.confidence)

        # Self-grown vocabulary: while talking, the mind occasionally coins
        # NEW sentence shapes of its own (topic-flavored, never verbatim
        # copies of what it just said) and keeps them in its phrase pool —
        # every conversation leaves it with more things it can say.
        self._grow_vocabulary(stimulus.text or "", cycle)

        # The language models learn from everything the mind reads AND says,
        # so their next-token predictions track the actual conversation.
        self._observe_language(stimulus.text)
        self._observe_language(response.text)
        # Self-improvement: this turn's text may grade the previous reply.
        self.improvement.observe_turn(self, stimulus.text or "", response, cycle)

        # emotional cooling happens after the cycle
        if self.flags.get("emotions", True):
            self.emotions.decay(0.08)
        return response

    # -- shortcuts -------------------------------------------------------------
    def ask(self, text: str) -> str:
        """Process a question and return just the reply text."""
        return self.process(Stimulus.text(text)).text

    def observe(self, text: str) -> Response:
        return self.process(Stimulus.text(text))

    def teach(self, text: str) -> List[Fact]:
        """Explicitly parse + store facts; returns what was learned."""
        facts = self.parser.parse(text)
        added: List[Fact] = []
        for fact in facts:
            if self.kb.add(fact):
                added.append(fact)
                self.events.emit("fact_learned", self.cycle_count, fact=str(fact))
            self.memory.add(str(fact), kind="fact", importance=0.7)
        self.goals.report_learning(len(added))
        self._observe_language(text)
        return added

    # -- training --------------------------------------------------------------
    def train(self, text: str) -> "TrainingReport":
        """Trains the mind on a piece of text — the train option for growing
        a mind's knowledge AND its vocabulary. Each line can teach three
        kinds of things:

          * facts ("birds are animals", "2 + 2 = 4") become knowledge,
          * a counting sequence ("0, 1, 2, ..., 9") becomes the number line,
          * plain conversational lines become SAYINGS — sentence shapes the
            mind may speak in its own voice later. The more sayings a mind
            owns, the more varied its replies and the less it repeats the
            same few sentences.

        Questions and conditional setups are requests, not knowledge or
        speech — they are skipped."""
        report = TrainingReport()
        for line in (text or "").split("\n"):
            self._train_line(line, report)
        return report

    def train_all(self, lines) -> "TrainingReport":
        """Bulk-trains the mind on a sequence of lines."""
        report = TrainingReport()
        for line in lines or []:
            self._train_line(line, report)
        return report

    def train_file(self, path: str) -> "TrainingReport":
        """Trains the mind from a corpus file: one utterance per line, blank
        lines and '#' comment lines are ignored."""
        with open(path, "r", encoding="utf-8") as fh:
            return self.train_all(fh.read().splitlines())

    def _train_line(self, raw: str, report: "TrainingReport") -> None:
        if raw is None or not raw.strip():
            return
        line = raw.strip()
        if line.startswith("#"):
            return  # comment line in corpus files
        report.lines += 1

        # 1. Knowledge: facts (plain or arithmetic) into the knowledge base.
        #    "facts_before" counts PARSED facts, not newly-added ones: a line
        #    whose fact we already know is still knowledge-shaped — it must
        #    never leak into the sayings vocabulary on re-training.
        facts_before = 0
        for fact in self.parser.parse(line):
            facts_before += 1
            if self.kb.add(fact):
                report.facts_learned += 1
                self.events.emit("fact_learned", self.cycle_count, fact=str(fact))
        if self.arithmetic.teach_sequence(line):
            report.sequences_taught += 1

        # 2. Vocabulary: a line that taught nothing and is not a request is
        #    conversational — it becomes a saying the mind can speak later.
        became_saying = facts_before == 0 and _is_sayable(line) and self.phrases.add(line)
        if became_saying:
            report.sayings_added += 1
            self.memory.add(line, kind="fact", importance=0.5)

        # Every training line also feeds the language models, so a trained
        # corpus directly improves the mind's language predictions.
        self._observe_language(line)

        self.emotions.apply({"joy": 0.08} if became_saying or facts_before else {})

    def feedback(self, reward: float) -> None:
        """Explicit user reward in [-1, 1]; consumed next cycle."""
        if self.learner is not None:
            self.learner.feedback(reward)
        self.events.emit("feedback", self.cycle_count, reward=reward)

    # ------------------------------------------------------------------ manual-only tools
    def search_web(self, query: str, max_results: int = 4,
                   focus: Optional[str] = None) -> str:
        """Explicit web search — the manual path to the internet.

        Runs the multi-engine chain (Google News → Wikipedia → Bing;
        DuckDuckGo was removed by design; a custom backend wins when
        assigned) and when it succeeds LEARNS from the results: snippets
        become recallable memories, clean "X is Y" sentences become facts
        (tagged "web"), the text feeds the next-token model, and the query's
        (and focus's) words count as known — she looked them up instead of
        asking.

        ``focus`` is the demo's suggestion argument — ``/searchweb "Flowers"
        Look for any information on flowers``: the quoted part stays the
        search query; the suggestion STEERS what she takes from the results
        (hits matching the focus words are learned and reported first).

        This is never fired by the cognitive cycle itself unless
        ``with_auto_web_fetch()`` turned the knowledge-gap lookup on — an
        explicit builder decision. Returns a formatted multi-line report (or
        a clean failure message)."""
        from ..capabilities.web_search import (
            MultiEngineBackend, WebSearchError, WebSearchFailure,
            rank_by_focus)
        if not query or not query.strip():
            return "No search query given."
        query = query.strip()
        try:
            results = self._run_web_search(query, max_results)
        except WebSearchError as exc:
            advice = {
                WebSearchFailure.OFFLINE:
                    "Web search needs an internet connection — check the "
                    "network and try again, or teach me directly.",
                WebSearchFailure.BLOCKED:
                    "The search engine sent a bot-check instead of results. "
                    "I can try again in a minute, or you can teach me directly.",
                WebSearchFailure.NO_RESULTS:
                    "The web answered, but nothing matched. Try different "
                    "words, or teach me directly.",
            }[exc.reason]
            return (f'I could not search the web for "{query}": {exc}. {advice}')
        if not results:
            return f'Nothing readable came back from the web for "{query}".'

        # The focus suggestion steers the report AND the learning order.
        if focus and focus.strip():
            results = rank_by_focus(results, focus.strip())
        lines = [f'Here is what I found on the web for "{query}"' +
                 (f' (focusing on: {focus.strip()[:60]})'
                  if focus and focus.strip() else "") + ":"]
        for i, r in enumerate(results, start=1):
            host = urllib.parse.urlparse(r.url).netloc if r.url else ""
            snippet = f" — {r.snippet[:140]}" if r.snippet else ""
            lines.append(f" {i}. {r.title}{snippet} ({host})")

        self._learn_from_web_results(query, results, focus)
        self.events.emit("web_search", self.cycle_count, query=query[:80],
                         results=len(results))
        return "\n".join(lines)

    def explore_url(self, url: str, focus: Optional[str] = None,
                    max_pages: Optional[int] = None) -> str:
        """Her own WebFetch-and-explore: reads a starting page and FOLLOWS its
        internal links (same host, focus-steered) so the mind is not locked
        to one page or one search engine — ``/explore <url> Look for any
        information on flowers`` reads the page plus related pages whose
        addresses/anchors match the suggestion. Everything read LEARNS: page
        texts become recallable memories (tag "web"), clean factual sentences
        become knowledge facts, and the text feeds the next-token language
        model. Focus words count as known afterwards. Manual by nature — only
        a caller (the demo command, a program) runs it."""
        from ..capabilities.web_fetch import WebExplorer
        from ..capabilities.web_search import WebSearchError, WebSearchFailure
        if not url or not url.strip():
            return "No web url given. Usage: /explore <url> [suggestion]."
        explorer = WebExplorer(
            max_pages=max_pages or self.explore_max_pages)
        try:
            pages = explorer.explore(url.strip(), focus)
        except WebSearchError as exc:
            advice = (WebSearchFailure.OFFLINE == exc.reason and
                      "Reading a page needs an internet connection — check "
                      "the network and try again." or
                      "That address did not look like a readable web page. "
                      "Try another url.")
            return f"I could not explore {url}: {exc}. {advice}"
        if not pages:
            return (f"I could not read anything at {url} — the site refused "
                    "or the page had no readable text. Try another url.")

        self._learn_from_pages(pages, focus)

        lines = [f'I explored "{pages[0].title}"' +
                 (f" and followed {len(pages) - 1} related page(s) inside "
                  "the site" if len(pages) > 1 else "") + ":"]
        for i, p in enumerate(pages, start=1):
            host = urllib.parse.urlparse(p.url).netloc if p.url else ""
            lines.append(f" {i}. {p.title} — {p.excerpt(160)} ({host})")
        lines.append("It is all in my memory now — ask me about it anytime.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ web helpers
    def _run_web_search(self, query: str, max_results: int):
        """Runs the configured backend; raises WebSearchError honestly."""
        from ..capabilities.web_search import MultiEngineBackend
        backend = self.web_search_backend or MultiEngineBackend()
        return backend.search(query, max_results)

    def _learn_from_web_results(self, query: str, results, focus) -> None:
        """Search results → memories, facts, language models, vocabulary."""
        # 1. Recallable memories — she "remembers" what she read online.
        for r in results[:3]:
            self.memory.add(f"Web: {r.title}. {r.snippet} (source: {r.url})",
                            kind="fact", importance=0.6, tags={"web"})
        # 2. Knowledge: clean factual sentences in the snippets become facts.
        added = 0
        for r in results:
            if added >= 3:
                break
            for fact in self.parser.parse(f"{r.title}. {r.snippet}"):
                fact.confidence = min(fact.confidence, 0.7)
                fact.source = "web"
                if self.kb.add(fact):
                    added += 1
                if added >= 3:
                    break
        # 3. Language + vocabulary: the text feeds next-token prediction, and
        #    the query's (and focus suggestion's) words count as known now.
        self._observe_language(" ".join(
            f"{r.title}. {r.snippet}" for r in results))
        if self.learner is not None:
            from ..util import tokenize as _tokenize
            for word in _tokenize(query):
                if len(word) > 2:
                    self.learner.lexicon.mark_spoken(word)
            if focus and focus.strip():
                for word in _tokenize(focus):
                    if len(word) > 2:
                        self.learner.lexicon.mark_spoken(word)

    def _learn_from_pages(self, pages, focus) -> None:
        """Explored pages → memories, facts, language models."""
        # 1. Recallable memories — she "remembers" what she read.
        stored = 0
        for page in pages:
            for i in range(0, len(page.text), 400):
                chunk = page.text[i:i + 400]
                self.memory.add(f"Web: {page.title}. {chunk} (source: {page.url})",
                                kind="fact", importance=0.6, tags={"web"})
                stored += 1
                if stored >= 4 or i + 400 >= len(page.text):
                    break
            if stored >= 4:
                break
        # 2. Knowledge: clean factual sentences in the page text become facts.
        if self.flags.get("reasoning", True):
            added = 0
            for page in pages:
                for fact in self.parser.parse(page.text):
                    fact.confidence = min(fact.confidence, 0.6)
                    fact.source = "web"
                    if self.kb.add(fact):
                        added += 1
                    if added >= 5:
                        break
                if added >= 5:
                    break
        # 3. Language + vocabulary.
        for page in pages[:2]:
            self._observe_language(page.text[:2000])
        if focus and focus.strip() and self.learner is not None:
            from ..util import tokenize as _tokenize
            for word in _tokenize(focus):
                if len(word) > 2:
                    self.learner.lexicon.mark_spoken(word)

    def _try_knowledge_gap_lookup(self, ctx) -> None:
        """The knowledge-gap auto lookup: unknown topic → search, learn,
        answer from what she found. A failed lookup stays silent."""
        from ..compose.composer import ResponseComposer
        from ..capabilities.web_search import WebSearchError
        text = ctx.text or ""
        lowered = f" {text.lower()} "
        if any(w in lowered for w in (" you ", " your ", " yourself ")):
            return                      # identity / personal — never the web
        if ResponseComposer._mentions_unknown_person(text):
            return                      # "who are they" is about the user's people
        topic = ResponseComposer._memory_topic(text)
        if not topic or topic == "that" or len(topic) < 3:
            return
        # Memory has ground ONLY when a recalled memory actually mentions the
        # topic — loosely related chatter about other things is not knowledge
        # ("hey Yume, what is a quokka?" must still look it up). The mind's
        # own conversation bookkeeping (topic-shift notes) does not count.
        if any(topic in rec.content.lower()
               for rec, _ in (ctx.recalled or [])
               if "conversation" not in rec.tags):
            return
        if topic in self._auto_lookup_topics:
            return                      # one lookup per unfamiliar topic
        self._auto_lookup_topics.add(topic)
        try:
            results = self._run_web_search(topic, 4)
        except WebSearchError:
            # Failed lookup stays SILENT: no hijacked reply — the normal
            # honest "teach me" composition runs unchanged.
            self._auto_lookup_topics.discard(topic)
            return
        self._learn_from_web_results(topic, results, None)
        summary = "; ".join(
            f'"{r.title}"' + (f" — {r.snippet[:120]}" if r.snippet else "")
            for r in results[:2])
        ctx.answer = ReasoningAnswer(
            text=(f"I don't know much about {topic} yet, so I searched the "
                  "web to understand it a bit more. Here is what I found: "
                  f"{summary}"),
            confidence=0.65, kind="auto-web-lookup")
        self.events.emit("tool_invoked", self.cycle_count,
                         tool="web-search", success=True)

    # ------------------------------------------------------------------ conversational helpers
    def _track_conversation_topic(self, stimulus: Stimulus) -> None:
        """Conversational understanding: tracks the talk's topic from cycle
        to cycle and remembers each shift (tag "conversation")."""
        if not self.conversational.understand:
            return
        if stimulus.modality not in (Modality.TEXT, Modality.MIXED):
            return
        text = stimulus.text or ""
        if not text.strip():
            return
        from ..compose.composer import ResponseComposer
        topic = ResponseComposer._memory_topic(text)
        if not topic or topic == "that":
            return
        prev = self._last_conversation_topic
        if prev and prev.lower() != topic.lower():
            self.memory.add(f"The conversation moved from {prev} to {topic}.",
                            kind="episode", importance=0.5,
                            tags={"conversation"})
            self.events.emit("conversation_shift", self.cycle_count,
                             **{"from": prev, "to": topic})
        self._last_conversation_topic = topic

    # why/how-style words that raise the automatic thinking effort
    _AUTOMATIC_WHY_WORDS = ("why", "how ", "explain", "imagine", "design",
                            "compare")

    def _choose_automatic_level(self, ctx) -> ThinkingLevel:
        """THINKING AUTOMATIC — the per-cycle effort choice, read straight
        off this cycle's facts; the ceiling clamps the result."""
        text = ctx.text or ""
        effort = 0.0
        if len(text) >= 140:
            effort += 1.0
        elif len(text) >= 45:
            effort += 0.5
        elif len(text) >= 15:
            effort += 0.3
        lowered = text.lower()
        if any(w in lowered for w in self._AUTOMATIC_WHY_WORDS):
            effort += 1.0
        if ctx.answer is None:
            effort += 0.4
        if ctx.derived_facts:
            effort += 0.5
        if not ctx.recalled:
            effort += 0.2
        values = self.emotions.snapshot().values()
        if values and max(values) > 0.5:
            effort += 0.25
        if effort < 0.7:
            level = ThinkingLevel.LOW
        elif effort < 1.4:
            level = ThinkingLevel.MEDIUM
        elif effort < 2.1:
            level = ThinkingLevel.HIGH
        elif effort < 2.8:
            level = ThinkingLevel.VERY_HIGH
        else:
            level = ThinkingLevel.MAX
        return self.thinking_options.ceiling.clamp(level)

    def self_improve(self) -> "SelfImprovementReport":
        """Runs one conscious self-improvement pass: retires sayings that
        consistently drew disapproval and reports everything the feedback
        ledger has learned so far."""
        return self.improvement.self_improve(self)

    def _observe_language(self, text: str) -> None:
        """Feeds a piece of text to BOTH language models — the n-gram counts
        it immediately, the transformer buffers it for its next (lazy)
        training pass, which runs automatically once enough accumulates."""
        self.language_model.observe(text)
        self.neural_language_model.observe(text)
        self.neural_language_model.auto_train()

    def predict_next(self, context: str, count: int = 5) -> List["TokenPrediction"]:
        """Next-token prediction: what would the mind say AFTER these words?
        A HYBRID: the n-gram model contributes exact local statistics, the
        transformer contributes generalization to unseen contexts (and while
        she has not trained her transformer yet, the n-gram answers alone)."""
        return _hybrid_predict(self.language_model, self.neural_language_model,
                               context, count)

    def score_fluency(self, text: str) -> float:
        """Blended fluency estimate for a candidate wording — how much it
        reads like the language the mind has absorbed (both models)."""
        return _hybrid_fluency(self.language_model, self.neural_language_model, text)

    def train_deep_model(self, epochs: int = 8) -> "TransformerTrainingReport":
        """Explicitly deep-trains the neural (transformer) language model on
        everything observed so far — the deep pass, with a full report."""
        return self.neural_language_model.train(epochs)

    def stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "name": self.name,
            "cycles": self.cycle_count,
            "facts": self.kb.size(),
            "memories": self.memory.size(),
            "goals_open": len([g for g in self.goals.goals if not g.done]),
            "capabilities": [c.name for c in self.capabilities],
            "personality": self.personality.to_dict(),
            "emotions": self.emotions.snapshot(),
            "thinking": self.thinking_options.to_dict(),
            "expression": self.expression.to_dict(),
        }
        if self.learner is not None:
            stats["learning"] = self.learner.stats()
        return stats

    # ------------------------------------------------------------------ gender
    @property
    def gender_identity(self) -> str:
        """The mind's gender identity word ("" when none was configured)."""
        return self.gender.identity

    @property
    def feeling(self) -> FeelingSnapshot:
        """How the mind feels RIGHT NOW — the developer-facing read of what
        chat no longer announces.

        Example::

            print(yume.feeling.current_status)
            # "feeling anticipation (intensity 0.15) - mood positive (+0.00)"

        Also exposes ``name`` / ``intensity`` / ``valence`` / ``arousal`` /
        ``mood`` for programmatic use. (The C# counterpart is
        ``yume.Feeling.CurrentStatus``.)
        """
        return FeelingSnapshot.from_state(self.emotions)

    @property
    def pronouns(self):
        """The pronoun set others use for the mind (UNSPECIFIED by default)."""
        return self.gender.pronouns

    def set_gender(self, identity: str, pronouns=None) -> None:
        """Sets the mind's gender identity at runtime — applies to
        describe_self() and 'who are you?' immediately, and persists with the
        next snapshot."""
        self.gender.set(identity, pronouns)

    def set_pronouns(self, pronouns) -> None:
        """Sets just the mind's pronoun set at runtime. Only she/her, he/him
        and UNSPECIFIED are allowed — UNKNOWN_PERSON (they/them) raises
        ValueError: they/them is reserved for describing unknown people."""
        from .gender import validate_self_pronouns
        candidate = pronouns or UNSPECIFIED
        validate_self_pronouns(candidate)  # rejected attempts change nothing
        self.gender.pronouns = candidate

    def gender_clause(self) -> str:
        """The gender clause used in self-description — "I am a girl, and I
        go by she/her pronouns." — or "" when no gender was configured."""
        return self.gender.gender_clause()

    def describe_self(self) -> str:
        """A first-person self-introduction: name, purpose, gender (when the
        mind has one) and current mood. Also the answer to 'who are you?'."""
        desc = getattr(self, "description", "") or ""
        if not desc:
            self_ = f"I am {self.name}, a mind built with CognifyPy"
        elif desc.lower().startswith(self.name.lower()):
            self_ = desc.rstrip(".!")
        else:
            self_ = f"I am {self.name} - {desc}"

        parts = [self_ + "."]
        clause = self.gender_clause()
        if clause:
            parts.append(clause)
        # The numeric feeling readout is developer-facing introspection —
        # shown in chat only when expression.show_feeling_reads is on. The
        # live state stays on the `feeling` property regardless.
        if self.expression.show_feeling_reads:
            parts.append(f"Right now {self.emotions.describe()}.")
        return " ".join(parts)

    # ------------------------------------------------------------------ snapshot
    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "version": "1.0.0",
            "name": self.name,
            "cycles": self.cycle_count,
            "flags": dict(self.flags),
            "personality": self.personality.to_dict(),
            "emotions": self.emotions.snapshot(),
            "emotion_sensitivity": self.emotions.sensitivity,
            "facts": self.kb.to_list(),
            "memories": self.memory.to_list(),
            "goals": self.goals.to_list(),
            "drives": {n: d.level for n, d in self.goals.drives.items()},
            "novel_seen": sorted(self._novel_seen),
            "capabilities": [c.name for c in self.capabilities],
            "sayings": self.phrases.all(),
        }
        if not self.gender.is_unset:
            data["gender"] = self.gender.to_dict()
        # Developer-facing options persist with the mind, so a restored mind
        # keeps its thinking level, tone and presentation switches.
        data["thinking_options"] = self.thinking_options.to_dict()
        data["expression"] = self.expression.to_dict()
        data["conversational"] = self.conversational.to_dict()
        data["web"] = {"allow": self.allow_web_search,
                       "auto_fetch": self.auto_fetch_web,
                       "explore_max_pages": self.explore_max_pages}
        if self.learner is not None:
            data["learning"] = self.learner.state_snapshot()
            data["encoder_dims"] = self.encoder.dims
            data["lexicon"] = self.learner.lexicon.to_dict()
        if self.language_model.is_trained:
            data["language_model"] = self.language_model.to_dict()
        if self.neural_language_model.is_trained:
            data["neural_language_model"] = self.neural_language_model.to_dict()
        data["improvement"] = self.improvement.to_dict()
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Mind":
        """Rebuild a Mind from a snapshot (tools must be re-registered)."""
        from ..capabilities.tool_use import ToolRegistry
        from .builder import MindBuilder
        personality = Personality.from_dict(data["personality"])
        mind = MindBuilder(data.get("name", "Cognify")) \
            .with_personality(personality) \
            .with_emotion_sensitivity(data.get("emotion_sensitivity", 1.0)) \
            .with_memory_capacity(max(10, len(data.get("memories", [])) + 20)) \
            .build()
        mind.flags = dict(data.get("flags", mind.flags))
        mind.cycle_count = int(data.get("cycles", 0))
        mind.emotions.load(data.get("emotions", {}))
        mind.kb.load(data.get("facts", []))
        mind.memory.load(data.get("memories", []))
        mind.goals.load(data.get("goals", []))
        for name, level in data.get("drives", {}).items():
            if name in mind.goals.drives:
                mind.goals.drives[name].level = clamp(float(level))
        mind._novel_seen = set(data.get("novel_seen", []))
        mind.phrases.restore(data.get("sayings", []))
        mind.gender = GenderOptions.from_dict(data.get("gender"))
        mind.thinking_options = ThinkingOptions.from_dict(data.get("thinking_options"))
        mind.expression = ExpressionOptions.from_dict(data.get("expression"))
        mind.composer.expression = mind.expression
        mind.conversational = ConversationalOptions.from_dict(
            data.get("conversational"))
        mind.composer.conversational = mind.conversational
        mind.composer.knowledge = mind.kb
        web = data.get("web") or {}
        mind.allow_web_search = bool(web.get("allow", mind.allow_web_search))
        mind.auto_fetch_web = bool(web.get("auto_fetch", mind.auto_fetch_web))
        mind.explore_max_pages = max(1, min(6, int(
            web.get("explore_max_pages", mind.explore_max_pages))))
        if "learning" in data and mind.learner is not None:
            dims = int(data.get("encoder_dims", 128))
            if dims != mind.encoder.dims:
                mind.encoder = HashingEncoder(dims=dims)
                mind.learner.encoder = mind.encoder
            mind.learner.load_snapshot(data["learning"])
        if data.get("lexicon") and mind.learner is not None:
            mind.learner.lexicon.load_dict(data["lexicon"])
        if data.get("language_model"):
            mind.language_model.load_dict(data["language_model"])
        if data.get("neural_language_model"):
            nd = data["neural_language_model"]
            # The snapshot carries the model's own architecture: rebuild with
            # THAT config so a mind trained on a bigger transformer stays on
            # it instead of silently shrinking to current defaults.
            try:
                from ..learning.transformer import TransformerConfig
                cfg = TransformerConfig(
                    dim=int(nd.get("dim", mind.neural_language_model.cfg.dim)),
                    heads=int(nd.get("heads", mind.neural_language_model.cfg.heads)),
                    layers=int(nd.get("layers", mind.neural_language_model.cfg.layers)),
                    context_len=int(nd.get("context_len", mind.neural_language_model.cfg.context_len)),
                    ffn_mult=int(nd.get("ffn_mult", mind.neural_language_model.cfg.ffn_mult)),
                    dropout=float(nd.get("dropout", mind.neural_language_model.cfg.dropout)),
                    learning_rate=float(nd.get("learning_rate", mind.neural_language_model.cfg.learning_rate)),
                    seed=int(nd.get("seed", mind.neural_language_model.cfg.seed)),
                    max_vocab=int(nd.get("max_vocab", mind.neural_language_model.cfg.max_vocab)))
                if (cfg.dim, cfg.heads, cfg.layers, cfg.context_len) != (
                        mind.neural_language_model.cfg.dim,
                        mind.neural_language_model.cfg.heads,
                        mind.neural_language_model.cfg.layers,
                        mind.neural_language_model.cfg.context_len):
                    mind.neural_language_model = TransformerLM(cfg)
            except (TypeError, ValueError):
                pass  # malformed config: keep the fresh default model
            mind.neural_language_model.load_dict(nd)
        mind.improvement.load_dict(data.get("improvement", {}))
        # ensure capabilities match snapshot list where possible
        wanted = set(data.get("capabilities", []))
        have = {c.name for c in mind.capabilities}
        if "tool_use" in wanted and "tool_use" not in have:
            from ..capabilities.tool_use import ToolUseCapability
            mind.capabilities.append(ToolUseCapability(ToolRegistry()))
        return mind

    # ------------------------------------------------------------------ helpers
    def _grow_vocabulary(self, text: str, cycle: int) -> None:
        """Occasionally coins a NEW topic-flavored sentence shape and keeps
        it in the phrase pool (never a verbatim copy of anything just said)."""
        if self.composer.rng.random() > 0.35 or self.learner is None:
            return
        from ..compose.composer import ResponseComposer
        topic = ResponseComposer._memory_topic(text) if text else ""
        if not topic or topic == "that":
            return
        key = f"grow:{topic}"
        if key in self._grown_from:
            return
        self._grown_from.add(key)

        fresh_word = self.learner.lexicon.fresh_word()
        shapes = [
            f"The more I think about {topic}, the more I want to know",
            f"I keep turning {topic} over in my mind - tell me more sometime",
            f"My notes on {topic} are growing - thank you for that",
            f"Something tells me {topic} is worth remembering",
        ]
        if fresh_word:
            shapes.append(f"Words like \"{fresh_word}\" are what make talking about {topic} interesting")
        pick = shapes[self.composer.rng.randrange(len(shapes))]
        template = ResponseComposer._template_key(pick)
        if not any(ResponseComposer._template_key(s) == template for s in self.phrases.all()):
            self.phrases.add(pick)

    def _caps(self, phase: str) -> List[Any]:
        return [c for c in self.capabilities if getattr(c, "phase", "perceive") == phase]

    def _identity_answer(self, text: str) -> Optional[ReasoningAnswer]:
        """Answer 'who are you' / 'introduce yourself' from the mind's identity
        (name, description, gender, mood) instead of treating it as an
        unknown topic."""
        t = (text or "").lower()
        if not any(p in t for p in (
                "introduce yourself", "who are you",
                "tell me about yourself", "who am i talking to")):
            return None
        return ReasoningAnswer(
            text=self.describe_self(),
            confidence=0.9,
            kind="describe",
        )

    def _similarity(self, a: str, b: str) -> float:
        if self._embedder is not None:
            return self._embedder.similarity(a, b)
        return 0.0

    def _novelty(self, text: str) -> float:
        tokens = set(tokenize(text))
        if not tokens:
            return 0.0
        known = set(self._novel_seen)
        for rec in self.memory.all():
            known.update(tokenize(rec.content))
        for f in self.kb.facts():
            known.update((f.subject, f.object))
        novel = len(tokens - known) / len(tokens)
        self._novel_seen.update(tokens)
        return clamp(novel)

    def _goal_relevance(self, text: str) -> float:
        goal = self.goals.active or self.goals.select_active()
        if goal is None:
            return 0.0
        goal_tokens = set(tokenize(goal.description))
        stim_tokens = set(tokenize(text))
        if not goal_tokens or not stim_tokens:
            return 0.0
        return clamp(len(goal_tokens & stim_tokens) / max(1, len(goal_tokens)))

    # -- dual-process helpers -------------------------------------------------

    def _contradicts_known(self, text: str) -> bool:
        """True when the statement NEGATES something the mind knows, or
        asserts something it has explicitly negated ("birds are not animals"
        vs. taught "birds are animals"). A violated expectation is the
        textbook surprise trigger."""
        if not text or not text.strip():
            return False
        try:
            facts = self.parser.parse(text)
        except Exception:
            return False
        for f in facts:
            if f.predicate == "is-not-a" and \
                    self.kb.get(f.subject, "is-a", f.object) is not None:
                return True
            if f.predicate == "is-a" and \
                    self.kb.contradicts(f.subject, "is-a", f.object) is not None:
                return True
        return False

    def _goal_style_bias(self, ctx: CognitiveContext, action: str) -> str:
        """The private goal biases WHICH STYLE feels right — a prosocial
        goal ("make the user smile") makes warm styles more likely without
        ever being spoken about. The goal is part of the mind's inner
        prompt; motivation is felt, never recited."""
        goal = ctx.active_goal
        if goal is None:
            return action
        if ctx.answer is not None and ctx.answer.confidence >= 0.65:
            return action          # a confident answer is never restyled
        text = (goal.description or "").lower()
        if any(w in text for w in ("smile", "kind", "comfort", "warm", "friend")):
            if action in ("trained", "curious", "share_thought"):
                return "emotional"
        if any(w in text for w in ("learn", "curios", "explore")):
            if action in ("trained", "share_thought"):
                return "curious"
        return action
