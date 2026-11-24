# Cognify — Full Documentation

**Cognify** (.NET 8) and **CognifyPy** (Python) are twin implementations of the same cognitive-architecture library: a `Mind` kernel surrounded by pluggable cognitive modules that let an AI **reason, think, learn, feel emotions, remember, pursue goals, carry a personality** — and optionally **see, hear, embed meaning, speak languages, use tools, write and analyze code, and generate media**. Both sides are behaviorally identical; every feature in this document exists in both languages unless a paragraph says otherwise.

This is the complete reference: every builder call, every runtime API, the full training guide (including the training loop and how to fix degenerate output like `negotiate negotiate negotiate …`), persistence, GGUF/safetensors export, serving with llama.cpp / Unsloth, and troubleshooting.

> Companions: `README.md` (feature tour, both repos) and `docs/DOCUMENTATION.md` (this file). CognifyPy names use `snake_case` with identical semantics; where the two differ mechanically, both spellings are shown.

## Contents

1. [Installation & quick start](#1-installation--quick-start)
2. [Core concepts](#2-core-concepts)
3. [Building a mind — MindBuilder reference](#3-building-a-mind--mindbuilder-reference)
4. [The Mind API — runtime reference](#4-the-mind-api--runtime-reference)
5. [Capability reference](#5-capability-reference)
6. [Training guide](#6-training-guide) — bulk training, the deep training loop, recipes, fixing gibberish
7. [Persistence](#7-persistence)
8. [Export — GGUF, safetensors, tokenizer, llama.cpp](#8-export--gguf-safetensors-tokenizer-llamacpp)
9. [Serving a mind with llama.cpp / Unsloth](#9-serving-a-mind-with-llamacpp--unsloth)
10. [Demos & CLI](#10-demos--cli)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)
12. [Project structure, tests, license](#12-project-structure-tests-license)

---

## 1. Installation & quick start

**Cognify (C#, .NET 8)** — requires the .NET 8 SDK; the only NuGet dependency is `Microsoft.Data.Sqlite`.

```bash
dotnet build Cognify.sln
dotnet test  tests/Cognify.Tests/Cognify.Tests.csproj
dotnet run --project demo/Cognify.Demo       # interactive showcase
dotnet run --project demo/Cognify.Yume       # "activate Yume" chat demo
```

Reference the library from your project (`dotnet add reference` to `src/Cognify/Cognify.csproj`), then:

```csharp
using Cognify.Core;

using var mind = MindBuilder.Create("Aria")
    .Describe("A curious, warm assistant")
    .WithThinking()                 // inner voice + chain of thought
    .WithReasoning()                // facts, rules, inference
    .WithEmotions()                 // 37 emotions + mood
    .WithMemory()                   // short-term / long-term / episodic
    .WithSelfLearning(l => l.Rate(0.1))
    .WithPersistence("aria.db")
    .Build();

Console.WriteLine(mind.Ask("what is a bird?"));
mind.Teach("birds are animals");
mind.Reinforce(0.9);                // reward the last reply — she learns
await mind.SaveAsync();
```

**CognifyPy (Python 3.10+)** — zero third-party runtime dependencies (standard library only).

```bash
cd CognifyPy
python -m pytest tests -q           # full test suite
python -m cognify                   # demo
python -m cognify.demo              # Yume walkthrough
```

```python
from cognify.core.builder import MindBuilder

with MindBuilder.create("Aria") \
        .describe("A curious, warm assistant") \
        .with_thinking() \
        .with_reasoning() \
        .with_emotions() \
        .with_memory() \
        .with_learning(rate=0.1) \
        .build() as mind:
    print(mind.ask("what is a bird?"))
    mind.teach("birds are animals")
    mind.feedback(0.9)
    mind.save("aria.db")            # snapshot restore: Mind.from_dict / load
```

A mind runs **fully offline, in-process**: no API keys, no GPU, no external AI services. Pluggable backend interfaces let you wire real models in when you want them (§5, §8).

---

## 2. Core concepts

### Mind

`Mind` is the cognitive kernel. It owns the configuration, the attached modules, all learned state (knowledge, memories, emotions, personality, weights) and the cognitive cycle. Thread safety: `Process` serializes access per mind (`ProcessAsync` offloads to the thread pool). Everything is seeded — identical builds behave identically — except emotion/memory dynamics, which use real time.

### Modules and the cognitive cycle

Every capability is an `ICognitiveModule` (Python: `cognify.capabilities.base`) with a `Name` and an integer `Order`. One cycle = one `Process(stimulus)` call; modules run in `Order` sequence and share a `CognitiveContext` workspace:

```
stimulus (text | image | audio | video)
   |
 personality (5)     traits shape exploration, deliberation, sensitivity
   |
 memory (10)         recall related memories, store the stimulus
   |
 vision (11)         scenes -> objects, colors, captions            [WithVision]
 audio (12)          sounds -> transcription, tempo, loudness       [WithAudio]
 multilingual (13)   language detection, translation                [WithMultilingual]
 embeddings (14)     semantic vectors, similarity search            [WithEmbeddings]
   |
 emotions (20)       OCC-style appraisal -> emotions, mood update
   |
 reasoning (30)      answer questions, forward-chain rules, induce patterns
   |
 goals (40)          pick focus goal, suggest action, track progress
   |
 tools (45)          invoke registered tools                        [WithTools]
 code (46)           generate scaffolds, analyze snippets           [WithCode]
 generation (47)     image/video specs, real WAV audio              [WithMediaGeneration]
   |
 thinking (50)       chain of thought; defers to perception/action modules
   |
 learning (60)       feature encoding, self-supervised valence training
   |
 response            MindResponse (content + confidence + full trace)
```

Perception/action modules may set the response first; thinking honors the **"first module wins"** contract and defers to them. The mind never parrots you: recalled memories are referenced obliquely, taught facts are acknowledged ("Got it — I just learned 1 new thing(s)") rather than echoed, and deliberation penalizes recently-used styles and previous-reply *templates*.

### Responses and the chain of thought

Each cycle returns a `MindResponse` (Python: `Response`) with `Content`, `Confidence`, the winning style, the cycle number and the full `ChainOfThought` (`response.Chain.Render()`), phases `Observe → Recall → Feel → Hypothesize → Evaluate → Generate → Deliberate → Conclude`. The live inner monologue streams on `mind.ThoughtProduced` and is inspectable any time via `mind.Thinking.LastThoughts` and `mind.Stream` (`ThoughtStream`).

### Events

| C# event | Python signal | Fired when |
|---|---|---|
| `CycleCompleted` | `on_cycle_completed` | a cycle produced its response |
| `EmotionChanged` | `on_emotion_changed` | an emotion's intensity moved |
| `ThoughtProduced` | `on_thought` | the thought generator produced a step |
| `MemoryRecalled` | `on_memory_recalled` | memories surfaced during a cycle |
| `GoalUpdated` | `on_goal_updated` | goals spawned / progressed / completed |
| `Learned` | `on_learned` | the learning engine trained (avg reward) |
| `ToolInvoked` | `on_tool_invoked` | a tool ran (name + `ToolResult`) |
| `MediaGenerated` | `on_media_generated` | a media artifact was created |

---

## 3. Building a mind — MindBuilder reference

The builder is **opt-in**: a mind has exactly the modules you attach. Capabilities: `reasoning`, `thinking`, `learning`, `emotions`, `memory`, `goals`, `personality`, `vision`, `audio`, `embeddings`, `multilingual`, `tools`, `code`, `generation`. Unknown capability names throw `MindConfigurationException`.

```csharp
MindBuilder.Create("ExampleAI")
    .WithAllModules()                     // everything …
    .Without("audio", "tools", "code", "generation")   // … minus four
    .Build();

// exactly three modules — nothing else:
MindBuilder.Create("ExampleAI")
    .WithThinking().WithSelfLearning().WithMemory().Build();
```

Python builders additionally offer `with_preset(...)`, `with_random_personality(seed)`, `only(*names)`, `with_tools(*funcs)` (register plain Python functions as tools) and `with_capability(Capability.X)`.

### Builder methods (C#)

| Method | Attaches / does |
|---|---|
| `Create(name)` / `Describe(text)` | name + self-description (used by identity answers) |
| `WithReasoning(cfg?, enabled: true, ceiling: ReasoningCeiling.Only5)` | knowledge base + inference engine (level, depth, trace budget, ceiling — its own `ReasoningLevel` ladder, rung for rung identical to thinking) |
| `WithThinking(cfg?, enabled: true, ceiling: ThinkingCeiling.Only5)` | chain of thought + thought generator (level, tone, budget, per-level budget ladder, automatic, ceiling) |
| `WithCognitiveLevel(level, ceiling: ThinkingCeiling.Only5)` | ONE effort level for both modules: sets the thinking level AND the reasoning level, each only when that module is enabled in this builder |
| `WithExpression(cfg)` | what the mind *shows* in chat (Feelings/FollowUps/Thinking/Meta switches) |
| `WithConversational(cfg?)` | talkativeness `rate` and how much it collects (`understand`) |
| `WithSelfLearning(cfg?)` | hybrid learning engine (MicroNet + style net + Q-table) |
| `WithEmotions(cfg?)` | 37-emotion appraisal engine + mood |
| `WithMemory(cfg?)` | short-term / long-term / episodic stores |
| `WithGoals(cfg?)` | drives + goal planner (`AddGoal("learn", priority: 0.8, "curiosity")`) |
| `WithPersonality(cfg?)` | Big Five traits (`Traits(openness:, conscientiousness:, …)`) |
| `WithGender(identity, pronouns?)` / `WithPronouns(set)` | identity + pronoun set (see §5.12) |
| `WithVision(cfg?)` | scene understanding (pluggable `IVisionBackend`) |
| `WithAudio(cfg?)` | sound classification + transcription (pluggable `IAudioBackend`) |
| `WithEmbeddings(cfg?)` | hashed n-gram embedder + semantic store |
| `WithMultilingual(cfg?)` | language detection + phrasebook translation |
| `WithTools(cfg?)` | tool registry + built-ins (clock, calculator, text-stats) |
| `WithWebSearch(cfg?)` | `web-search` tool — **manual-only**, explicit `SearchWeb` calls |
| `WithAutoWebFetch(cfg?)` | `web-fetch` tool (URL reader, auto-enabled flows) |
| `WithCode(cfg?)` | code generation + static analysis |
| `WithMediaGeneration(cfg?)` | image/video specs + real WAV speech/audio synthesis |
| `WithModule(module)` | attach any custom `ICognitiveModule` |
| `WithPersistence(path, autoSaveEveryCycles: 0)` | SQLite store; `Build()` creates the DB immediately |
| `WithAllModules()` / `Without(params names)` | bulk enable / strip (case-insensitive) |
| `Lenient(bool lenient = true)` | degrade gracefully instead of throwing on disabled capabilities |
| `MindBuilder.Restore(path)` | static: rebuild a previously persisted mind from its SQLite file |

Runtime attach/detach: `mind.AttachModule(...)` / `mind.DetachModule<T>()` (Python: `mind.attach_module(...)` / `mind.detach_module(...)`); remaining modules keep working.

### Lenient vs strict

Strict is the default so misconfigurations stay loud: calling `See` without vision, `Hear` without audio, `InvokeTool` without tools throws `MindConfigurationException`. With `.Lenient()` the same calls degrade:

| Call | Without module, in lenient mode |
|---|---|
| `See(desc)` / `Hear(content)` | full cognitive cycle on the text (note lands in `mind.Stream`) |
| `Teach(sentence)` | plain memory, returns 0 facts learned |
| `Reinforce(reward)` | no training, but the mind still feels the reward |
| `InvokeTool(name)` | `ToolResult.Fail(...)` instead of an exception |
| `GenerateMedia(...)` | `null` |
| `DetectLanguage(text)` | `null` |
| `SemanticSimilarity(a, b)` | `0` |

Check `mind.Lenient` (Python: `mind.lenient`) at runtime.

---

## 4. The Mind API — runtime reference

Conversation and teaching:

| C# | Python | What it does |
|---|---|---|
| `mind.Process(stimulus)` | `mind.process(...)` | one full cognitive cycle; accepts a string or a `Stimulus` (kind: text/image/audio/video) |
| `mind.Ask(text)` | `mind.ask(text)` | process + return the reply content |
| `mind.Observe(text)` | `mind.observe(text)` | feed text to the language models without demanding a reply |
| `mind.Teach(sentence)` | `mind.teach(text)` | parse one line into facts; returns the learned `Fact`s |
| `mind.Train(text)` / `TrainAll(lines)` / `TrainFile(path)` | `train` / `train_all` / `train_file` | bulk training — the four-bucket teacher (§6.1) |
| `mind.Reinforce(reward)` | `mind.feedback(reward)` | reward the last response, −1..1 (learning + feeling) |
| `mind.SelfImprove()` | `mind.self_improve()` | review pass over the implied-feedback ledger; retires disliked sayings |
| `mind.Tick()` | `mind.tick()` | advance time: drives grow, emotions decay, moods drift |

Perception, action and knowledge access:

| C# | Python | Module |
|---|---|---|
| `mind.See(desc)` / `mind.Vision.LastScene` | `see` / `vision.last_scene` | vision |
| `mind.Hear(content)` / `mind.Hearing.LastHeard` | `hear` / `hearing.last_heard` | audio |
| `mind.SemanticSimilarity(a, b)` / `mind.SemanticMemory.FindSimilar(t, topK)` | `semantic_similarity` / `semantic_memory.find_similar` | embeddings |
| `mind.DetectLanguage(text)` / `mind.Languages.Translator` | `detect_language` / `languages.translator` | multilingual |
| `mind.InvokeTool(name, args?)` / `mind.Tools.Registry.Register(tool)` | `invoke_tool` / `tools.registry.register` | tools |
| `mind.SearchWeb(query)` | `search_web(query, max_results)` | web search (manual-only) |
| `mind.Knowledge` (`AddRule(LogicRule)`, facts) | `knowledge` (`add_rule`) | reasoning |
| `mind.Memory.Remember(...)` / `Recall(q)` | `memory.remember` / `memory.recall` | memory |
| `mind.GoalSystem.AddGoal(...)` | `goal_system.add_goal` | goals |
| `mind.EmotionalState` / `mind.Feeling.CurrentStatus` | `emotional_state` / `feeling.current_status` | emotions |
| `mind.GenerateMedia(prompt, type)` | `generate_media(prompt, kind)` | media generation |
| `mind.PredictNext(ctx, n)` / `mind.ScoreFluency(text)` | `predict_next` / `score_fluency` | next-token |
| `mind.ApplyChatTemplate(messages, template?, tools?, addGenerationPrompt?, extraVariables?)` | `mind.apply_chat_template(messages, template?, tools?, add_generation_prompt?, **kwargs)` | render a llama.cpp-compatible Jinja chat template (§9) |
| `mind.TrainDeepModel(epochs \| options)` | `train_deep_model(epochs \| options)` | deep training loop (§6.2) |
| `mind.NeuralLanguageModel` / `mind.Learning` / `mind.Lexicon` / `mind.Phrases` | `neural_language_model` / `learning` / `lexicon` / `phrases` | learning internals |
| `mind.SaveAsync()` / `mind.LoadFromAsync(path)` / `MindBuilder.Restore(path)` | `save(path)` / `Mind.from_dict(...)` | persistence (§7) |
| `mind.Config.Thinking.At(...)` / `.UpTo(...)` / `.Enable(false)` | `mind.config.thinking...` | runtime thinking reconfiguration |
| `mind.SetGender(...)` / `mind.SetPronouns(...)` / `mind.DescribeSelf()` | `set_gender` / `set_pronouns` / `describe_self` | gender/identity |

`Stimulus` (Python: `Stimulus` in `cognify.core.stimulus`) wraps content with a `StimulusKind` (`Text`, `Image`, `Audio`, `Video`) so `See`/`Hear` and `Process` share one path.

---

## 5. Capability reference

### 5.1 Reasoning — logic and knowledge
A symbolic knowledge base of `Fact`s (subject-predicate-object) and variable-carrying `LogicRule`s with three inference modes: **deduction** (forward chaining / modus ponens), **induction** (co-observed concepts become `related-to` facts), **abduction** (best rule + antecedents explaining an observation). Facts are also learned from ordinary sentences every cycle, and are stored and spoken in natural words — teach "Before is the affect before the cause" and "what is before?" yields *"Before is affect before cause."*, never a hyphenated slug. Letter-skeleton comparison makes spaced forms and legacy slugs compare equal; leading fillers ("so what is X?") are skipped before question matching.

**Reasoning levels + trace budget, lined up with thinking** — reasoning has its OWN ladder: `ReasoningLevel` (`Low` … `Max`) with `ReasoningCeiling` (`Only1` … `Only5`), enums that mirror the thinking ones rung for rung (cross-map with `level.FromThinking()` / `level.ToThinking()`; Python: `ReasoningLevel.from_thinking(...)` / `.to_thinking()`, `ReasoningCeiling.from_thinking(...)`), and the same default budgets (50/100/200/300/400). The level sizes two things: the forward-chaining depth — `Low` 2, `Medium` 6 (the historical default), `High` 8, `VeryHigh` 10, `Max` 12 passes — and the reasoning TRACE budget, the tokens of arithmetic / inference / learning explanations one cycle may spend (`TokensFor(level)` / `tokens_for(level)`; a global `Budget(n)` overrides the ladder, per-rung `Budget(ReasoningLevel.High, 260)` sets one rung, `Depth(n)` pins an explicit depth the ladder stops retuning). One call sets the effort of both modules: `.WithReasoning().WithThinking().WithCognitiveLevel(ThinkingLevel.High)` thinks at High AND chains at High (Python: `with_reasoning(level="high")` or `with_cognitive_level("high")`; runtime: `mind.set_reasoning_level("high")`, `mind.set_reasoning_budget(120, "high")`). The budgeted trace is visible, not hidden: its surviving lines surface in the chain of thought as `Reasoning: …` steps (Hypothesize phase, capped at four) alongside thinking's own stages — with thinking disabled at build time there is no chain, and the trace still rides on `response.Reasoning.Explanations`. The spend is reported on every response (`ReasoningResult.BudgetUsed` / `BudgetLimit`; Python: `ctx.reasoning_trace` and the `reasoning.budget.*` scratch keys). Facts are never dropped by the budget — only the visible trace is trimmed.

```csharp
mind.Teach("a bird can fly");
var answer = mind.Ask("what is a bird?");      // "bird is a animal. bird can fly. ..."
mind.Knowledge.AddRule(new LogicRule(
    new[] { new FactPattern("?s", "is-a", "bird"), new FactPattern("?s", "is-in", "cage") },
    new FactPattern("?s", "can-not", "fly")));
```

**Taught arithmetic** — the mind learns math from teaching and reasons with it: exact lookup, counting on via taught "n + 1" facts, larger taught jumps, counting along a taught number line ("the system is 0, 1, 2, …, 9"), commutativity, subtraction backwards, multiplication as repeated addition, division as repeated subtraction, plus step-by-step word problems ("no apples and i buy an apple" → 1 via the zero identity). Wrong results ("2 + 2 is 5") are refused and never learned. Anything off the taught line is admitted honestly — or, with tools attached, transparently answered by the calculator ("I have not been taught that one yet, so I used the calculator: …"). Arithmetic questions route to the calculator only as a fallback; plain questions are never hijacked and questions are never parsed into facts.

### 5.2 Thinking — the inner voice
The thinking module builds an inspectable chain of thought and generates grounded thoughts (topic, feeling, recalled memories, inferred facts, new vocabulary) before any reply is composed.

- **Level** — how many thoughts per cycle: `Low` = 1, `Medium` = 2 (default), `High` = 4, `VeryHigh` = 6, `Max` = 8; or an explicit token budget `Budget(300)` (≈ budget/50 thoughts). Every level keeps its promise even on a bare "hello".
- **Per-level token budgets** — a budget LADDER, one entry per level from Low up to Max (or up to the ceiling's top level when a clamp is set): `Budget(ThinkingLevel.High, 260)` (Python: `budget(260, "high")` or `mind.set_thinking_budget(260, "high")`). The chosen level's budget sizes the thought set (≈ budget/50 thoughts, 1..10), so re-leveling re-sizes it. Defaults reproduce the classic ladder (50/100/200/300/400 → 1/2/4/6/8). Reads clamp — `TokensFor(ThinkingLevel.Max)` on an Only2-capped model answers with Medium's budget — and writes clamp too, so a Max budget on an Only2 model lands on Medium. A single global `Budget(300)` overrides every per-level entry (and suppresses automatic level choice — you named the exact budget). `tokens <= 0` removes a rung, falling back to the default ladder. Budgets serialize with the config; `TokensFor(level)` / `tokens_for(level)` reads them back at any time.
- **Automatic** — `WithThinking(t => t.Auto())` (Python: `with_thinking(automatic=True)`) lets the mind choose its effort per turn: questions (+2), why/how probes, long inputs, novel ground, emotional surges and fresh epistemic gaps raise the level; the choice is clamped by the ceiling and recorded in the chain ("Automatic: chose to think at High"). With a ceiling set, the resolver only picks the LEVEL — the chosen level's own token budget follows. Hosts that reason outside the loop call the standalone resolver directly: `ThinkingAuto.Choose(text, hasDirectAnswer, recalledCount, emotionIntensity, epistemicGap, ceiling)` (Python: `cognify.thinking.thinking_auto.choose(...)`) — deterministic, same weights, never exceeds the ceiling.
- **Ceiling** — the hard cap on how far the model may *ever* think: `ThinkingCeiling.Only1` (Low only) … `Only5` (everything, default). Requests above the ceiling — build-time or runtime (`mind.Config.Thinking.At(...)`) — are silently clamped to the ceiling's top level. The cap travels into exported GGUF (`cognify.thinking.ceiling` next to `cognify.thinking.enabled` / `cognify.thinking.level`).
- **Tone** — `"balanced"`, `"focused"`, `"playful"`, `"poetic"`, `"plain"`.
- **Switch** — `WithThinking(enabled: false)` (or `mind.Config.Thinking.Enable(false)`) makes the model answer directly with no thoughts at all.
- **Lines up with reasoning** — reasoning owns its own `ReasoningLevel` / `ReasoningCeiling` enums, rung for rung identical to the thinking ones, with the same default budgets: `WithCognitiveLevel(ThinkingLevel.High)` maps the level 1:1 onto both modules in one call (or level each independently: `WithThinking(t => t.At(ThinkingLevel.High))` + `WithReasoning(r => r.At(ReasoningLevel.Max))`), and the exported GGUF names the levels in llama.cpp's `--reasoning-effort` vocabulary (`low`, `medium`, `high`, `xhigh`, `max` — VeryHigh maps to `xhigh`), with each module's per-level budget ladder next to them.

```csharp
.WithThinking(t => t.At(ThinkingLevel.High)      // 4 thoughts per cycle
                     .WithTone("poetic")
                     .Budget(ThinkingLevel.Max, 480)   // the ladder, per rung
                     .UpTo(ThinkingCeiling.Only3))  // Max requests clamp to High
```

Deliberation generates candidates in seven styles (informative, empathetic, curious, concise, reflective, encouraging, trained) and picks one with the hybrid learned value model; fluency from the language models breaks close ties.

The conversation itself is tracked separately from the options: `new ConversationTracker()` (Python: `ConversationTracker()` from `cognify.core.conversation`). Feed each turn with `tracker.Observe(mind, userText, replyText, cycle)` (Python: `tracker.observe(mind, user_text, reply_text, cycle)`) and read `tracker.Topics` / `PendingQuestions` / `Familiarity` / `Summary()` (Python: `topics` / `pending_questions` / `familiarity` / `summary()`). Topics are counted from the user's content words; a user question whose reply shares none of its content words is kept as an open thread (max 3, fading after 8 more cycles); every 6th turn the top topics land in long-term memory ("We have been talking about …"), so the understanding survives save/reload like the rest of the mind.

### 5.3 Expression & conversational knobs — what the mind shows
Chat stays natural by default: feelings are *felt* (an epistemic feeling comes out as "Hm — that one caught me off guard"), internals are invisible unless the host surfaces them. `WithExpression(e => e.Feelings(true).FollowUps(false).Thinking(true).Meta(true))` restores mechanical feeling announcements, disables her own follow-up questions, or lets the host show the thought trace / confidence-style-cycle meta line. `WithConversational(r => r.rate(2.0))` (Python: `with_conversational(rate=2.0)`) scales talkativeness. Follow-up asks ("Who are they to you?") are on by default; her honest reaction to brand-new words is not a follow-up and stays available either way.

### 5.4 Self-learning — the hybrid engine
Six cooperating mechanisms: (1) **MicroNet**, a real feed-forward net (hashing → tanh → sigmoid) trained with backprop to predict reward/valence; (2) **NeuralStyleEngine**, a style-conditioned net (text features ⊕ style one-hot ⊕ mood) predicting how rewarding each style is here and now; (3) **reinforcement** — `mind.Reinforce(reward)` updates a per-style Q-table with personality-driven ε-greedy exploration; (4) **adaptive blending** — final score `(1−w)·symbolic + w·neural`, with `w` (0.25–0.80) drifting with the style net's accuracy ("hybrid blend: 52% neural / 48% symbolic" appears in the chain); (5) **self-supervised** valence training from the cycle's own appraisal; (6) **experience replay** via `learning.Replay()`. Stats: `mind.Learning.Stats`, `mind.Learning.NeuralWeight`.

### 5.5 Vocabulary growth — the Lexicon
The lexicon listens to every message, picks up never-heard words, weaves them back into conversation ("I just picked up the word \"axolotl\" from you — what does it mean in your world?") and coins its own topic-flavored sayings into the phrase pool while talking. Words already woven into a reply are never asked about again; everything survives save/restore. API: `mind.Lexicon.Size` / `RecentNewWords()` / `LastPicks` (Python: `lexicon.size` / `recent_new_words()` / `last_picks`).

### 5.6 Next-token prediction — n-gram + transformer
A trigram language model with stupid backoff observes everything the mind reads and says (user turns, replies, taught facts, trained corpora, web results). It powers fluency-aware deliberation (mean per-token backoff log-probability breaks close ties) and literal predictions:

```csharp
var next = mind.PredictNext("birds are", 3);    // hybrid: exact n-gram + neural generalization
string completion = mind.NeuralLanguageModel.Generate("birds are", maxTokens: 8);
```

Untrained minds score 0 — the fluency term is inert until the mind has absorbed text. The model persists (SQLite `langmodel` / `deeplangmodel`) and merges on restore.

### 5.7 Deep learning — the neural stack
From-scratch, dependency-free, pure C#/Python:

| Piece | What it is |
|---|---|
| `MicroNet` | one-hidden-layer feed-forward net (the self-learning engine's neural half) |
| `DeepNet` | genuinely deep MLP: arbitrary hidden layers, ReLU/GELU/Tanh, He init, inverted dropout, L2 weight decay, mini-batch **Adam**, softmax+cross-entropy or sigmoid/linear+MSE |
| `TransformerLM` | decoder-only GPT-style transformer: token + learned positional embeddings, pre-LayerNorm blocks of multi-head causal self-attention + GELU FFNs, residual stream, weight-tied output head, gradient clipping, temperature sampling — backward pass hand-derived |
| `HybridNextToken` | the blend: n-gram exactness + transformer generalization; an untrained transformer degrades gracefully to pure n-gram |
| `AdamOptimizer` | the shared Adam implementation |

`TransformerConfig` defaults (both sides): `Dim = 32`, `Heads = 4`, `Layers = 2`, `ContextLen = 16`, `FfnMult = 4`, `Loop = 1`, `Dropout = 0.1`, `LearningRate = 3e-3`, `Seed = 42`, `MaxVocab = 2048`, `BatchSize = 16`, `MaxWindowsPerEpoch = 512` (Python: 96), `GradClip = 1.0`. The transformer **trains lazily** — automatically once ~600 new tokens (`AutoTrainTokens = 600`, `AutoTrainEpochs = 6`) have accumulated, or explicitly via `mind.TrainDeepModel(...)`. Weights persist in the snapshot; when trained they also export as `cognify.transformer.*` GGUF/safetensors tensors (§8).

**The Loop Transformer** (`TransformerConfig.Loop`, 1–8, default 1) — weight-tied recurrent depth (Universal Transformer / ALBERT style): the block stack is applied `Loop` times per forward pass, so `Loop = 2` with `Layers = 1` runs ONE block's weights twice — the same parameter count, twice the effective depth. The model "thinks longer" per token without growing, and gradients sum over every application so looping trains exactly like a deeper stack with tied weights. Configure it through the builder (`WithTransformer` / `with_transformer`); snapshots carry it (a one-pass model refuses a looped snapshot and vice versa), and the llama.cpp export unrolls it into `Layers × Loop` physical blocks — an unrolled model computes exactly the same function, and llama.cpp has no weight tying across blocks (`gpt2.block_count` reports the unrolled count; `cognify.transformer.layers` / `cognify.transformer.loop` record the logical stack, §8).

```csharp
var report = mind.TrainDeepModel(epochs: 20);
Console.WriteLine(report);   // "20 epoch(s) on 512 window(s): loss 4.1 -> 1.9 (21,856 parameters, 96 words, 812 ms)"

var net = new DeepNet(16, [32, 32], 4, DeepActivation.GELU, DeepOutput.Softmax, dropout: 0.1);
net.Train(X, classLabels, epochs: 100, learningRate: 0.01);
var lm = new TransformerLM(new TransformerConfig { Dim = 48, Layers = 3, ContextLen = 24 });
lm.Observe("some training text"); lm.Train(epochs: 30);

// Loop Transformer: one 1-block stack, looped twice — same parameters as
// Layers = 1, twice the effective depth.
using var loopedMind = MindBuilder.Create("Fino")
    .WithMemory()
    .WithTransformer(c => c with { Layers = 1, Loop = 2 })
    .Build();
```

### 5.8 Self-improvement — learning from reactions
Implied feedback: the user's next message grades the previous reply ("good job / thanks" → reward; "no, that's wrong / stop repeating" → disapproval; questions never count as grades), credited to the style and wording actually spoken. Every reply template keeps an approval EMA — disliked wordings are demoted, approved ones promoted. `mind.SelfImprove()` runs an explainable review pass that **retires** trained sayings whose score stays clearly negative; `mind.SelfImprovement.RecordFeedback(reply, -1.0)` grades explicitly. The ledger persists (`selfimprovement` table) and rides on the learning capability.

### 5.9 Emotions — 37 feelings + mood
Plutchik's eight primaries plus 29 finer Darwin-taxonomy feelings (love, devotion, tenderness, suffering, weeping, high spirits, low spirits, anxiety, grief, dejection, despair, hatred, disdain, contempt, guilt, pride, helplessness, patience, affirmation, negation, self-attention, shyness, modesty, blushing, reflection, meditation, ill-temper, sulkiness, determination). OCC-inspired appraisal maps events to emotions via keyword desirability, expectation violation and reward signals; **social triggers** are research-based (blame/unfairness → anger, startle markers → surprise, shown interest → anticipation, negation of known facts → violated-expectation surprise). Emotions blend, decay over `Tick()`, aggregate into mood (valence + arousal). `EmotionalState.Describe()` → "love (0.96), joy (0.62) | mood positive (+0.18)"; direct control via `Feel(EmotionType.Surprise, 0.5)`. PascalCase names are humanized (`SelfAttention` → "self attention").

**Dual-process impulse gate** (Zajonc's affect primacy): a surge above `EmotionOptions.ImpulseThreshold` (default 0.55), or blame/dismissal/contradiction, blurs a first-person reaction ("Wait — WHAT? I did not see that coming at all!") instead of a deliberated reply, returning to normal as the surge decays. Confident knowledge (arithmetic, taught facts, identity) is never blurted over. **Epistemic questions**: surprise/interest spikes open a knowledge gap and the mind asks ("What is X, exactly? You have me curious now.") when the user isn't already asking.

### 5.10 Memory — three stores + associative network
Short-term (fixed-capacity recency), long-term (importance + rehearsal, strength decays without use), episodic (events tagged with the emotion felt then; intense ones become non-decaying flashbulb memories). Recall combines keyword overlap over **content words**, a spreading-activation associative network (co-occurring concepts get linked) and salience. Guards: near-identical restatements rehearse the existing memory instead of stacking copies, the current stimulus is never echoed back as a memory, and unrelated messages can't hijack recall.

```csharp
mind.Memory.Remember("the vault code is 4471", Cognify.Memory.MemoryKind.LongTerm, importance: 0.95);
var hits = mind.Memory.Recall("what was the vault code?");   // finds it
```

### 5.11 Goals — intrinsic motivation
Four drives (**curiosity, social, achievement, calm**) grow over time; pressing drives autonomously spawn goals. Goals carry priority, progress, deadlines and urgency-based focus selection, and are **private motivation**: a prosocial goal biases HOW the mind talks but is never recited. `mind.GoalSystem.AddGoal("finish-report", priority: 0.9, tags: ["achievement"], deadline: DateTimeOffset.UtcNow.AddDays(2))`; forty `Tick()`s later the mind has new goals of its own.

### 5.12 Personality & gender — character and identity
Big Five (OCEAN) traits shape processing: neuroticism amplifies negative emotions, extraversion amplifies joy, openness drives exploration, conscientiousness deepens deliberation and goal focus, agreeableness warms responses and softens anger. Gender: `WithGender("girl", PronounSet.Female)` gives an identity word + pronoun set; exactly two presets exist — `Female` (she/her) and `Male` (he/him) — plus genderless `Unspecified` (first-person only). `PronounSet.UnknownPerson` (they/them) is **reserved for unknown other people** and rejected for the mind itself; the mind naturally uses they/them when talking about "a friend" or "someone". Gender survives save/restore. `mind.DescribeSelf()`, `mind.SetGender("boy", PronounSet.Male)`, `mind.SetPronouns(PronounSet.Female)`.

### 5.13 Vision — eyes
`mind.See(...)` (or an image/video stimulus) runs a pluggable `IVisionBackend`. The built-in `SimulatedVisionBackend` turns scene descriptions into `VisualScene`s: objects with confidences and frame positions, dominant colors, brightness, scene type (indoor/outdoor/portrait/document) and a caption. Swap in a real model via `WithVision(v => v.Backend = myBackend)`.

### 5.14 Audio — ears
`mind.Hear(...)` (or an audio stimulus) runs a pluggable `IAudioBackend`: classifies clips as speech/music/noise/tone, estimates duration/loudness/tempo, and produces transcriptions for speech that become the cycle's effective content. Loud sounds trigger startle (surprise). `WithAudio(a => a.Backend = myWhisperBackend)` plugs a real recognizer.

### 5.15 Embeddings — a sense of meaning
`HashedNGramEmbedder` is a real embedding model: signed feature hashing of word unigrams, bigrams and character trigrams, L2-normalized. The `EmbeddingStore` keeps capacity-bounded semantic memory with cosine search — "this new thing resembles something I've seen" even with zero keyword overlap. `mind.SemanticSimilarity("the cat sat on the mat", "a cat sat on a mat")` ≈ 0.47; unrelated pairs ≈ 0. Vectors persist with the mind; `IEmbeddingModel` is pluggable.

### 5.16 Multilingual — languages
Two-stage dependency-free `LanguageDetector`: Unicode script analysis (Han, Kana, Hangul, Cyrillic, Arabic, Devanagari, incl. Ukrainian-vs-Russian and Japanese-with-Kanji disambiguation), then stopword/signature-word profiling for English, Spanish, French, German, Portuguese, Italian, Dutch. The `Phrasebook` translates common phrases (greetings, thanks, yes/no, politeness) and takes custom phrases; `AutoTranslate` converts foreign input into the preferred language before deeper processing. Optional real backend: `mind.Languages.Translator = myApiTranslator`.

### 5.17 Tools — hands
A `ToolRegistry` of `ITool`s (name, description, parameters, trigger phrases), natural-language argument extraction and invocation inside the cycle. Built-ins: **clock** (UTC date/time), **calculator** (real tokenizer + recursive-descent parser: `+ - * / %`, parentheses, unary minus, `1,000` grouping — no `eval`), **text-stats** (word/sentence/character counts, reading time). With `WithWebSearch()` a fourth, **web-search**, ships — see below.

```csharp
mind.Ask("what time is it?");              // "It is currently 23:38:56 UTC on ..."
mind.Ask("Hello Yume, what is 2 + 2?");    // math found anywhere in the sentence
mind.Tools.Registry.Register(new MyWeatherTool());
var result = mind.InvokeTool("weather", new Dictionary<string, string> { ["city"] = "Oslo" });
```

### 5.18 Web search & web fetch — internet, explicit only
The `web-search` tool carries **no trigger phrases and is excluded from auto-invocation**: the mind never goes online unprompted. The only path is explicit — `mind.SearchWeb("what is a quokka")` (Python: `mind.search_web(...)`, demo `/searchweb`). Successful searches are *learned from*: snippets become memories, clean "X is Y" sentences become facts (source "web"), text feeds the next-token model, and the query's words count as known. Failures (offline, blocked) say so cleanly and suggest teaching directly; searches retry once with a timeout. Own backend: `Tools.WebSearchBackend = myIWebSearchBackend`. `WithAutoWebFetch()` adds the URL-reading `web-fetch` tool (`mind.ExploreUrl(url, focus?)` / `explore_url`).

### 5.19 Code — compiler's eye
Generation: `TemplateCodeBackend` emits scaffolds (class, function, loop, SQL query, endpoint) in C#, Python, JavaScript or SQL with derived identifiers and doc comments; `ICodeBackend` is pluggable for real LLM generation. Analysis: `CodeAnalyzer` does real static analysis — line/comment counts, bracket balance, nesting depth, complexity score, frequent tokens, language detection; fenced markdown blocks are detected automatically. `mind.CodeEngine.Analyze(code).Summarize()`.

### 5.20 Media generation — a studio
`MediaGenerationModule` recognizes generation intents ("generate an image of…", "create a video of… for 10 seconds", "say hello", "make a sound of…") and produces `MediaArtifact`s via a pluggable `IMediaBackend`. Images/video are deterministic specs derived from a prompt hash (same prompt = same artifact). Speech/audio is a **real, playable WAV**: 16-bit PCM, RIFF header, tone bursts per syllable with pitch variation, envelopes and click-free fades. `mind.GenerateMedia("hello world", MediaType.Speech)`; gallery via `mind.MediaGeneration.Gallery`; events via `mind.MediaGenerated`.

### 5.21 Extend it yourself
Implement `ICognitiveModule` (or derive from `CognitiveModuleBase`) and `WithModule(...)` it. Custom modules can produce responses too — earlier modules win; thinking defers to them.

---

## 6. Training guide

There are two training systems, and they answer different needs:

| | Bulk trainer (`Train` / `Teach`) | Deep training loop (`TrainDeepModel`) |
|---|---|---|
| What it changes | knowledge (facts, arithmetic) + sayings (sentence shapes) | the transformer's weights (next-token prediction) |
| Input | curated lines | any text the mind observed (corpus, conversation) |
| Deterministic? | yes — parse & store | no — gradient descent on a loss |
| Affects exported GGUF? | facts only via training data | **yes — the weights themselves** |

A mind needs both: bulk training gives it things to *say and know*; deep training gives its neural half the ability to *predict and generalize* text. Neither replaces the other.

### 6.1 Bulk training — the four-bucket teacher

`Train` sorts every line into four buckets:

| Line kind | Example | Becomes |
|---|---|---|
| Fact | `Birds are animals.` | knowledge-base fact (reasoning) |
| Dictionary line | `Joy means a bright feeling of gladness.` | `(joy, means, …)` fact — recallable via "what does joy mean?" |
| Arithmetic / number line | `1 + 1 = 2.` / `0, 1, 2, …, 9` | taught math (counting ladder; no calculator needed) |
| Plain conversational line | `I find the stars endlessly fascinating.` | **saying** — a sentence shape the mind can speak in its own voice |

Questions and conditional setups ("if i have no apples…") are requests, not knowledge — skipped. Training never duplicates: re-feeding the same corpus adds nothing the second time. `#` comments are skipped in files.

```csharp
var report = mind.Train("I enjoy quiet mornings.");        // single utterance
var report = mind.TrainAll(File.ReadLines("corpus.txt"));  // bulk
var report = mind.TrainFile("corpus/yume-starter.txt");    // file
Console.WriteLine(report);   // "43 line(s): 23 fact(s), 19 saying(s), 1 number line(s)."
```

`TrainingReport` fields: `Lines`, `FactsLearned`, `SayingsAdded`, `SequencesTaught`. Sayings feed the dedicated `trained` response style (word-overlap scoring, rotation, recent-skip) and persist in snapshots. `mind.Phrases.Count` / `Phrases.All()` show the spoken vocabulary. Dictionary lines make meanings *recallable knowledge*: after training the shipped `corpus/words-and-meanings.txt` (1,127 words), all four phrasings work — `what does joy mean?`, `what is the meaning of grief?`, `define patience`, `what is a fox?`.

### 6.2 The deep training loop — `TransformerTrainOptions`

`mind.TrainDeepModel(epochs)` trains the transformer with sensible defaults. For full control pass a `TransformerTrainOptions` (C#) / `TransformerTrainOptions` dataclass (Python):

```csharp
var report = mind.TrainDeepModel(new TransformerTrainOptions {
    Epochs       = 40,
    LearningRate = 0.03,     // per-run override; null keeps the config's 3e-3
    Decay        = 0.95,     // LR multiplied by this after EVERY epoch (1.0 = constant)
    Patience     = 4,        // stop after 4 consecutive stagnant epochs (0 = never)
    MinDelta     = 1e-4,     // an epoch counts as progress when loss improves ≥ this (nats/token)
    Progress     = (e, total, loss) => Console.WriteLine($"epoch {e}/{total}: {loss:F3}"),
});
```

```python
report = mind.train_deep_model(TransformerTrainOptions(
    epochs=40, learning_rate=0.03, decay=0.95, patience=4, min_delta=1e-4,
    progress=lambda e, total, loss: print(f"epoch {e}/{total}: {loss:.3f}"),
))
```

**Every knob, exactly:**

| Option | Default | Meaning |
|---|---|---|
| `Epochs` / `epochs` | 12 (C#) / 8 (Py) | epochs planned for this run (≥ 1) |
| `LearningRate` / `learning_rate` | `null`/`None` | learning rate for THIS run; null keeps `TransformerConfig.LearningRate` (3e-3) |
| `Decay` / `decay` | `1.0` | **a multiplier, not a percent**: LR is multiplied by this after every epoch. `1.0` = constant; `0.95` = −5% per epoch; `0.9` = −10% per epoch. Setting it to `90` multiplies the LR by 90 each epoch — the loss explodes immediately |
| `MinDelta` / `min_delta` | `0.0` | minimum loss improvement (nats/token) that counts as progress toward early stopping |
| `Patience` / `patience` | `0` | consecutive non-progressing epochs allowed before the run stops early; 0 disables early stopping |
| `Progress` / `progress` | `null` | callback fired after every epoch: `(epoch, epochsPlanned, loss)` |

`TransformerTrainOptions.For(epochs)` (C#) builds a plain run of the given length. The plain overload `TrainDeepModel(12)` is unchanged and behaves exactly as before.

**The report** (`TransformerTrainingReport`, both sides) carries `Epochs` (`epochs`), `Windows` (`windows`), `InitialLoss`/`FinalLoss` (`initial_loss`/`final_loss`), `Parameters` (`parameters`), `Vocabulary` (`vocabulary`), `Duration` (`duration`), and the loop extras `PerEpochLosses` (`per_epoch_losses`), `LearningRates` (`learning_rates`) and `StoppedEarly` (`stopped_early`) — so the report can render the full loss curve and LR schedule. `ToString()` summarizes: `"40 epoch(s) on 512 window(s): loss 4.102 -> 0.913 (21,856 parameters, 96 words, 3,204 ms) [stopped early] "`.

**Lazy auto-training** is unchanged: the transformer observes everything the mind reads and says, and trains automatically once ≥ 600 new tokens (`AutoTrainTokens`) have accumulated since the last pass (`AutoTrainEpochs = 6`). `lm.ShouldAutoTrain` / `lm.AutoTrain()` expose it; scripted or small corpora should call `TrainDeepModel` explicitly, as the automatic pass only fires on real accumulation. `lm.Observe(text)` feeds text without demanding a reply.

### 6.3 Recipe — how to train a mind that talks

**The single most important rule: corpus quality IS reply quality.** The transformer is word-level and tiny; it can only recombine the words its corpus contained. Every reported case of gibberish — *"things gear more welcome gear pressure it"*, *"negotiate negotiate negotiate …"* — has the same root cause: **the model was trained (or served) on a corpus that doesn't contain the conversation**.

The concrete case: the shipped dictionary corpus (`corpus/words-and-meanings.txt`) is 1,127 lines of `Word means …`. It contains zero occurrences of "hello" — but it does contain *gear*, *pressure* and *negotiate*. A mind deep-trained only on it has no greeting behavior to learn; asked "hello", its undertrained word-level model falls back to looping a frequent corpus word — hence `negotiate negotiate negotiate …`. This is not a bug in the trainer; it is the corpus missing the behavior you want.

The fix, in order:

1. **Feed conversational lines.** `greeting-corpus.txt` (ships with both repos' download sets) is the minimal example: dozens of natural greeting exchanges (`hello`, `hi`, `good morning`, `how are you`, `talking starts when you say hello`, …). Any corpus that contains the *kind of exchange* you want works the same way.
2. **Deep-train with the loop — and hold out a validation split.** A small corpus wants more epochs and early stopping, and `TrainFile` can hold out 20% of the file so you can *watch* generalization instead of guessing:

   ```csharp
   // 80% of the file trains, 20% is held out and scored (deterministic
   // split — the same file + seed always hold out the same lines).
   var report = mind.TrainFile("greeting-corpus.txt", validationSplit: 0.2);
   Console.WriteLine(report);   // "... Validation: 9 line(s) held out, loss 1.234 nats/token ..."
   mind.TrainDeepModel(new TransformerTrainOptions {
       Epochs = 40, LearningRate = 0.03, Decay = 0.95, Patience = 4, MinDelta = 1e-4,
       Progress = (e, total, loss) => Console.WriteLine($"epoch {e}/{total}: {loss:F3}"),
   });
   var reply = mind.Ask("hello");   // -> "Talking starts when you say hello"
   ```

   The report's `ValidationLoss` is the SAME cross-entropy the deep-training loop minimizes (nats/token), computed forward-only on the held-out lines — directly comparable to the epoch losses, and NaN until the deep model has been trained once. If validation loss goes UP while training loss goes down, the model is memorizing, not learning. Validate first, then run one plain `TrainFile(path)` pass so the final model has seen everything. (Python: `mind.train_file(path, validation_split=0.2)`; `mind.neural_language_model.evaluate(lines)` scores any lines without touching the weights.)

   `scripts/diagnose_hello.py` (CognifyPy repo scripts set) reproduces the whole story end-to-end: train on the dictionary alone → "hello" yields word salad; add the greeting corpus + loop → "hello" yields a real reply.
3. **Mix your corpora.** Dictionary lines teach recallable definitions; conversational lines teach speech. A mind that only ever saw definitions can only speak definitions. Keep both, plus facts and a number line, and dedup is automatic.
4. **Then export.** A served GGUF speaks with its raw transformer alone — no phrase pool, no reasoning, no lexicon. Whatever the weights learned is all the listener gets. Train first, export second.
5. **Serve with the exported sampling profile.** The same trained weights can look brilliant or broken depending on the sampling settings: a tiny model served at `temperature 1.61` with `Repetition Penalty: Off` and `Max Tokens: Max` loops and rambles for thousands of tokens per reply even when the corpus is right. The export writes the recommended values as `cognify.sampling.*` GGUF metadata (§9) — set them in llama.cpp's flags or Unsloth's sampling panel (`--temp 0.7 --top-k 40 --top-p 0.9 --repeat-penalty 1.2 --n-predict 128` for a greeting-scale model) before judging the output.

**Never wrap `TrainFile` in a re-train loop.** Facts are deduplicated, but the language buffer is not: every pass re-feeds every line into the next-token counts, so looping `TrainFile` 20 times over-weights the corpus's most frequent bigrams 20-fold. The symptom is exactly the *"good morning is warm"* class of reply — two corpus fragments ("good morning", "…is warm") blended by raw bigram frequency, with no grammar to hold them apart. One `TrainFile` pass (with a validation split) plus `TrainDeepModel` epochs is the whole training recipe; if you want more language exposure, add more corpus lines, not more passes over the same lines. The runtime's own generation now carries the same defenses the serve flags give llama.cpp: sampling includes the end token (a finished sentence can end itself instead of running to the token cap) and a no-triple guard re-samples any word that would land three times in a row.

**Hyperparameter starting points** (the tiny default transformer: 32-dim, 2 layers, 16-token context):

| Setting | Recommended | Why |
|---|---|---|
| `LearningRate` | `0.03` (or `null` for the 3e-3 default) | 0.1 and above make the tiny model's updates overshoot — loss bounces instead of descending. If you raise it, raise `Decay` decay too |
| `Decay` | `0.95` (gentle) or `0.9` | smooths the descent; **always < 1.0 and > 0.1 — it is a multiplier, not a percent** |
| `Epochs` | 30–60 for small corpora; 10–20 for large ones | small corpora need repetition; large ones don't |
| `Patience` | 4–6 with `MinDelta = 1e-4` | stops the run when the loss stalls, saving time and overfitting |
| `MinDelta` | `1e-4` (fine) — avoid `1e-2` | `0.01` nats/token is a huge improvement for a tiny model; with a coarse MinDelta the run declares "no progress" while the loss is actually still falling |

**Reading the run:** per-epoch losses should descend smoothly (a sawtooth means the LR is too high); `learning_rates` should show the decay; `stopped_early = true` with a flat loss tail means the model extracted what it could — more epochs would add nothing. For real headroom, grow the architecture via `TransformerConfig` (`Dim`, `Layers`, `Heads`, `ContextLen`) — at the cost of slower training.

### 6.4 Training checklist for a served model

1. Build the mind with the capabilities you want (they travel as GGUF metadata, §8).
2. `TrainFile(...)` your corpora: facts, dictionary, **conversational lines**, arithmetic.
3. `TrainDeepModel(TransformerTrainOptions { … })` until the loss curve flattens.
4. Sanity-check: `mind.Ask("hello")`, `mind.PredictNext("good", 3)`, a couple of taught questions.
5. Export (`MindExporter.ExportGgufForLlamaCpp`, §8) and re-check the served model.

---

## 7. Persistence

```csharp
.WithPersistence("aria.db", autoSaveEveryCycles: 25)   // DB created at Build() time
await mind.SaveAsync();                                 // full snapshot
await mind.LoadFromAsync("aria.db");                    // reload into the same mind
using var revived = MindBuilder.Restore("aria.db");     // one-line restore
```

The database is created the moment the mind is built — schema plus initial state — so `Restore(path)` works even before the first explicit save, and every cycle keeps using the same file (auto-saving when configured). The SQLite store persists: profile, config, emotion intensities, mood, all memories (strength/recall counts), facts, rules, goals, drives, the trained neural networks (MicroNet + style net), the Q-table, the associative network, the semantic (embedding) memory, the n-gram model (`langmodel`), the **transformer** — weights, vocabulary, training buffer and architecture (`deeplangmodel`) — the lexicon and its spoken words, the self-improvement ledger (`selfimprovement`) and the sayings pool. Python offers `mind.save(path)` / `Mind.from_dict(...)` over the same state.

---

## 8. Export — GGUF, safetensors, tokenizer, llama.cpp

`Cognify.Export.MindExporter` (Python: `cognify.export.exporter`) writes standard ML interchange formats:

```csharp
var files = MindExporter.ExportBundle(mind, "export/yume");
// yume.gguf, yume-llamacpp.gguf, yume.safetensors, tokenizer/tokenizer.json,
// train_config.json, finetune.sh, quantize.sh, training_data.txt, val_data.txt,
// config.json, README.md

MindExporter.ExportGguf(mind, "yume.gguf");                     // GGUF v3, cognify architecture (archive)
MindExporter.ExportGgufForLlamaCpp(mind, "yume-llamacpp.gguf"); // RUNNABLE gpt2 GGUF  <- for llama.cpp/Unsloth
MindExporter.ExportSafetensors(mind, "yume.safetensors");       // Hugging Face tensors
MindExporter.ExportTokenizer(mind, "tokenizer/");               // HF WordPiece tokenizer.json + vocab.json
```

Export APIs take a **file** path; an existing directory target writes the file inside it named after the mind, and missing parents are created (no raw `UnauthorizedAccessException`). The runnable GGUF needs a **trained** transformer — call `mind.TrainDeepModel(...)` explicitly before exporting unless real conversation has already triggered the lazy pass.

**The llama.cpp-runnable GGUF** maps the transformer onto the **stock `gpt2` architecture**: exact llama.cpp tensor names (`token_embd.weight`, `position_embd.weight`, `blk.N.attn_norm`, `blk.N.attn_qkv` fused as GPT-2's `c_attn`, `blk.N.attn_output`, `blk.N.ffn_up`, `blk.N.ffn_down`, `output_norm`; weight-tied output head omitted), a full `tokenizer.ggml` set (byte-unicode `Ġ` tokens, i32 token types, special ids, BPE merges) so prompt encoding reaches the word ids she trained with, and **post-write self-verification**: every file is re-read, compared value-for-value and linted against the gpt2 loader conformance rules — a broken export fails at export time, not inside llama.cpp. A Loop Transformer mind exports **unrolled**: `Layers × Loop` physical `blk.N` blocks (llama.cpp has no weight tying across blocks; an unrolled model computes exactly the same function), with `gpt2.block_count` = the unrolled count and `cognify.transformer.layers` / `cognify.transformer.loop` recording the logical stack it came from.

**Context length and the "Total x / 16" readout** — `gpt2.context_length` equals the trained window (`TransformerConfig.ContextLen`, default 16: the learned absolute position embeddings bake it into the weights). `ExportOptions.LlamaCppContextLength = 128` stretches it by linear position interpolation (pi-style), so Unsloth/llama.cpp show /128 and `llama-server -c 128` works; coherence degrades the further past the trained range you push. Per-completion length is a request-time knob, not a weight: `n_predict` on the command line, `max_tokens` in the OpenAI-compatible API / Unsloth's max-tokens field.

**Capability metadata + chat template** — every builder switch travels twice. First as `cognify.*` KV metadata: `cognify.modules`, one `cognify.<capability>.enabled` boolean per known module (audio, code, tools, reasoning, memory, emotions, personality, goals, vision, embeddings, multilingual, self_learning, media_generation), `cognify.web_search.enabled` / `cognify.web_fetch.enabled`, `cognify.tools.count` / `cognify.tools.names`, `cognify.description`, and the effort metadata for both cognitive modules — `cognify.thinking.{enabled,level,ceiling,budget}` and `cognify.reasoning.{enabled,level,ceiling,budget}`, the level names in llama.cpp's `--reasoning-effort` vocabulary (`low`, `medium`, `high`, `xhigh`, `max`; VeryHigh maps to `xhigh`) so a client can pass them straight to the server, and `budget` the token cap the current level buys (llama.cpp `--reasoning-budget` semantics: N>0 = token cap per cycle). Second as a JSON manifest embedded in `tokenizer.chat_template` (a Jinja `{%- set cognify_manifest = '…' %}` that renders nothing; `llama-server` exposes the template via `GET /props`, so clients can extract it and **gray out what the mind lacks**): the `thinking` and `reasoning` objects each carry `enabled`, the operating `level`, the hard `ceiling`, the implemented effort ladder (`levels`, lowest→highest — the same shape as llama.cpp's `chat_template_caps.reasoning_effort_levels`) and the per-level token budgets (`budgets`, e.g. `{"low": 50, …, "max": 400}`), so a client can drive both modules' levels exactly the way `--reasoning-effort` / `--reasoning-budget` do. The template also honors Qwen3's `enable_thinking` kwarg: `chat_template_kwargs: {"enable_thinking": false}` drops the `assistant: thought:` cue (thinking minds end on it; non-thinking minds carry no cue). `cognify-chat.html` (repo download set) is a ready-made client for `llama-server --jinja`: capability chips, grayed disabled modules, a Unsloth-style "Thinking · <level>" selector with above-ceiling options locked, tools popover, tok/s, reasoning_content blocks.

**What a GGUF cannot carry** — a GGUF is frozen weights and llama.cpp is a bare inference runtime: the modules behind `WithAudio()`, `WithCode()`, `WithTools()`, web search, the thinking loop, memory and the sayings pool execute only in the Cognify runtime. The `cognify.*` keys *document* what the mind carries; the chat template only *formats prompts*. Facts can be baked into weights by training on them; everything else needs the runtime or an agent harness around the server (§9).

**Sharded safetensors** — `ExportSafetensors(mind, "yume.safetensors", 62)` writes `yume-00001-of-00006.safetensors …` + `yume.safetensors.index.json` (`metadata.total_size` + `weight_map`), shards balanced by payload size and always holding whole tensors, reassembly byte-for-byte identical. `ExportOptions.SafetensorsSplit` applies the same sharding inside `ExportBundle`.

**Diagnosing any GGUF** — `GgufDiagnostics.Inspect(path)` (Python: `cognify.export.diagnostics`) parses GGUF v2/v3 — including files from other tools — and reports which quantization types it actually contains against llama.cpp's ggml type table, plus a verdict and next steps. If a GGUF refuses to load with "tensor layout does not match this build's quantization types", the file carries quant types newer than the installed build (dynamic quants from publisher forks, MXFP4) — pick another quant, use the named build, or re-quantize an F32 source with `llama-quantize`.

What each format carries:

| File | Contents |
|---|---|
| `*.gguf` | GGUF v3, custom `cognify` architecture: llama.cpp-style metadata + F32 tensors `cognify.net.*` (micro-net) + RL `cognify.rl.*` +, when trained, the transformer's `cognify.transformer.*`. Spec-compliant (gguf-py / llama-quantize read it) but not runnable by llama.cpp (no graph for it) |
| `*-llamacpp.gguf` | Runnable `gpt2` GGUF of the trained transformer — **the file for llama.cpp / Unsloth / llama-cpp-python** |
| `*.safetensors` | The same tensors in HF format (`safetensors.numpy.load_file`) |
| `tokenizer/tokenizer.json` | HF WordPiece tokenizer from the mind's own corpus, `[PAD] [UNK] [BOS] [EOS] [MASK]` specials |
| `train_config.json` | llama.cpp finetune-style hyperparameters |
| `finetune.sh` / `quantize.sh` | Ready-to-run `llama-finetune` (LoRA) and `llama-quantize` → Q4_K_M / Q5_K_M / Q8_0 pipelines |
| `training_data.txt` / `val_data.txt` | The mind's corpus, ~90/10 split |

`ExportOptions` tunes the bundle (model name, vocab size, quant type, context/batch/epochs/lr/threads); tune the mind itself via `MindBuilder` before exporting. Honest scope: a Cognify mind is a cognitive architecture, not an LLM-scale model — the exports are real, spec-compliant artifacts (training seeds, LoRA bases, weight snapshots, runnable tiny LMs), and the `-llamacpp` GGUF is verified end-to-end with llama.cpp, llama-cpp-python and Unsloth.

---

## 9. Serving a mind with llama.cpp / Unsloth

1. Build + train the mind (§6.4), then `ExportGgufForLlamaCpp(mind, "fino-llamacpp.gguf")` with `ExportOptions.LlamaCppContextLength` set to the window you want Unsloth to display.
2. Serve: `llama-server -m fino-llamacpp.gguf -c 128 --jinja` (`-c` is the runtime context; the UI's context usage tracks it; `--jinja` enables the chat template so `enable_thinking` works).
3. Point `cognify-chat.html` at the server for capability chips / thinking selector / tools UI, or any OpenAI-compatible client at `http://localhost:8080/v1`.
4. **Runtime bridge for real capabilities**: tools, web search, audio, code execution and the thinking loop are Cognify runtime modules. Two supported patterns: (a) run the mind in-process for cognition and use llama.cpp as its generation backend; (b) wrap the llama.cpp server in an agent harness that executes tools and re-prompts. The GGUF metadata + chat template exist so the *UI layer* knows what the model supports; no metadata can give llama.cpp new abilities.

Unsloth note: exported models load with the bundled word-level tokenizer; out-of-vocabulary prompt words fall back to `<unk>`, and the FFN uses the tanh GELU approximation while llama.cpp evaluates exact GELU (sub-0.003 delta per activation).


### Recommended sampling settings — the exported `cognify.sampling.*` metadata

Every GGUF export ships the sampling profile the model needs to produce clean output, sized to its vocabulary:

| keys | < 1k vocab pieces (greeting / toy scale) | < 32k pieces (small chat model) | bigger |
| --- | --- | --- | --- |
| `cognify.sampling.temperature` | 0.7 | 0.7 | 0.8 |
| `cognify.sampling.top_p` | 0.9 | 0.9 | 0.9 |
| `cognify.sampling.top_k` | 40 | 40 | 40 |
| `cognify.sampling.repeat_penalty` | 1.20 | 1.15 | 1.10 |
| `cognify.sampling.max_tokens` | 128 | 256 | 512 |

Why: a word-level model trained on a small corpus can only recombine the words it saw, and it has no learned sense of when to stop. At a hot temperature with the repetition penalty OFF it falls into "what you is what you is what you" loops and rambles until the token limit — thousands of tokens of noise per reply. The profile tames the sampling, breaks the loops and caps the reply.

Serving flags for a small Cognify model (Unsloth: same values in its sampling panel):

```
llama-server -m fino-llamacpp.gguf -c 16384 --jinja \
  --temp 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --n-predict 128
```

Reading the generation stats: speed (`tok/s`) and draft acceptance (`Draft accepted 99%`) are the serving stack doing its job — nothing to optimize there. `First token: 1.60s` is the draft-model warm-up, not the model's quality. The number that matters is `Tokens:` per reply — if it is in the thousands for a one-line answer, the sampling settings (temperature / repeat penalty / max tokens) are wrong, not the model.


### Chat templates — the built-in Jinja engine (llama.cpp compatible)

Both runtimes carry a dependency-free mini-Jinja engine (`Cognify.Core.ChatJinja` /
`cognify.core.jinja`) that renders the templates llama.cpp's `minja` accepts — the
subset real model templates (ChatML, Qwen2.5, Qwen3, Llama-3.1, Zephyr) actually
use. `Mind.ApplyChatTemplate` / `apply_chat_template` is the entry point:

```csharp
var prompt = mind.ApplyChatTemplate(
    new object?[] { ("user", "Weather in Tokyo?") },
    template: File.ReadAllText("qwen3.jinja"),    // any llama.cpp --jinja template
    tools: tools,                                 // OpenAI-style dictionaries
    addGenerationPrompt: true,
    extraVariables: new Dictionary<string, object?> { ["enable_thinking"] = false });
```

```python
prompt = mind.apply_chat_template(
    [("user", "Weather in Tokyo?")],
    template=Path("qwen3.jinja").read_text(),
    tools=tools, add_generation_prompt=True,
    enable_thinking=False)   # extra kwargs become template variables
```

With no template, the mind's OWN GENERATED template is used (below).
`bos_token` / `eos_token` default to the mind's special tokens like llama-server
does; messages may be role/content dictionaries or `(role, content)` tuples.

### Generated templates — the mind writes its own

The chat template shipped in a GGUF is not a hand-written constant: each mind
SYNTHESIZES it from its live state. `mind.GenerateChatTemplate(manifestJson?)` /
`mind.generate_chat_template(manifest_json=None)` composes a minja-compatible
template from what the mind actually carries:

- a provenance comment naming the mind, its thinking switch/level/ceiling,
  capabilities, tool names and description — renders nothing, but is
  extractable from GET /props next to the manifest, and every value is
  sanitized so a description can never terminate the comment early;
- the manifest `set` line — only in the variant the exporter embeds;
- a tools branch (`{% if tools %}tools: {{ tools | tojson }}{% endif %}`) —
  ONLY for minds with the tools capability: a plain mind's template cannot
  even express tools, matching the capability the GGUF documents;
- the `role: content` message loop and a thinking-aware assistant cue —
  the Qwen3 `enable_thinking` kwarg handling exists only for minds that
  can think.

Generation is deterministic (same mind state → same template bytes), and
`Mind.ApplyChatTemplate` / `apply_chat_template` default to the mind's own
generated template, so the runtime format and the exported file agree by
construction. The exporter embeds exactly `mind.GenerateChatTemplate(...)`,
tags the file with `cognify.chat_template.generator = cognify-builtin-jinja/1`,
and SELF-VERIFIES the generated template through the built-in engine before
writing it — a template that cannot render the mind's own format byte-exactly
aborts the export instead of shipping broken.

Supported (mirrors minja): `{{ expr }}` output; `{% if/elif/else %}`; `{% for %}`
with the `loop` variable (`index/index0/first/last/length/revindex`) and
for-`else`; `{% set %}` with plain, tuple and `ns.attr` targets — assignments
inside a loop stay in that iteration's scope, which is why templates use
`namespace()` to accumulate state; `{% generation %}` blocks (llama.cpp
extension); `{# comments #}`; `-` whitespace control on every tag; filters
(`tojson` with sorted keys like llama.cpp's nlohmann dump plus `indent`, `trim`,
`length`/`count`, `first`, `last`, `join`, `items`, `reject`/`select` with
tests, `default`, `upper`/`lower`/`capitalize`, `replace`, `reverse`, `string`,
`list`, `int`, `float`, `sort`); tests (`defined`/`undefined`, `none`, `string`,
`number`, `integer`/`float`, `boolean`, `true`/`false`, `mapping`, `iterable`,
`equalto`/`eq`, `ne`, `divisibleby`); the global functions `raise_exception`,
`strftime_now`, `namespace`, `range`; slices with negative steps
(`messages[::-1]`, `messages[1:]`); and string/dict methods (`split`, `strip`
family, `startswith`/`endswith`, `replace`, `get`, `keys`, `values`, `items`).
Not supported (clear errors, like minja without its callback hooks): macros,
includes, extends/blocks, `{% raw %}`, the `map`/`selectattr` filters and
auto-escaping.

Compatibility is pinned by fixtures, not by hope: real templates fetched from
HuggingFace (`tests/templates/*.jinja`, provenance in its README) render to
committed golden bytes (`tests/goldens/`) that BOTH engines must reproduce
byte-for-byte — the C# and Python suites fail if the two implementations ever
drift. The GGUF exporter's `tokenizer.chat_template` is emitted by the same
shared emitter the engine renders (`ChatJinja.CognifyChatTemplate` /
`cognify_chat_template`), so an exported file's chat format, the capability
manifest extractable from /props, and the runtime default can never disagree.

---

## 10. Demos & CLI

**Cognify.Demo** (showcase): `/teach <fact>`, `/feel <emotion> <0-1>`, `/reward <number>`, `/see <scene>`, `/hear <words>`, `/similar <text>`, `/generate <Image|Video|Speech|Audio> <prompt>`, `/state`, `/goals`, `/memories`, `/save`, `/exit`.

**Cognify.Yume** (activation walkthrough):

```bash
dotnet run --project demo/Cognify.Yume -- --train corpus/yume-starter.txt
dotnet run --project demo/Cognify.Yume -- --train corpus/words-and-meanings.txt
```

Chat commands: `train <file>`, `teach <sentence>`, `sayings`, `vocab`, `gender <identity>`, `pronouns <set>`, `who are you`, and the control slash-commands `/teach`, `/train <file>`, `/searchweb <q>` (explicit only), `/improve`, `/predict <text>`, `/trainbrain [epochs]`, `/brain`, `/feeling`, `/feelings on|off`, `/followups on|off`, `/thinking on|off`, `/showthinking on|off`, `/level <low|medium|high|veryhigh|max>`, `/ceiling <only1..only5|1..5>`, `/tone <…>`, `/meta on|off`, `/help`, `/exit`. Startup switches: `--thinking <level>`, `--thinking-ceiling <…>`, `--thinking-tone <…>`, `--no-thinking`, `--show-feelings`, `--show-internals`, `--no-followups`. CognifyPy's demo (`python -m cognify.demo`) ships the same corpora (`corpus/words-and-meanings.txt`, 1,127 words) and the same command set.

---

## 11. Troubleshooting & FAQ

**"My model says `negotiate negotiate negotiate …` (or other word salad) when I greet it."**
Not a training-loop bug — a corpus bug. The word-level transformer can only recombine words its corpus contained; a dictionary corpus with zero "hello" teaches no greeting behavior, and an undertrained model loops a frequent corpus word. Fix: train on conversational lines (see `greeting-corpus.txt`) and deep-train with the loop (§6.3). `scripts/diagnose_hello.py` demonstrates the before/after. The same rule applies to a served GGUF: corpus quality is reply quality.

**"Why does my mind repeat the same sentences?"**
Spoken variety comes from the sentence shapes it owns. Train sayings (50+ makes repetition statistically rare), teach facts so answers come from knowledge, teach arithmetic/number lines, keep chatting (memories + lexicon growth add shapes). Deliberation already penalizes recent styles, previous replies and their templates.

**"Unsloth shows `Total 6 / 16`."**
16 is the trained context window baked into the position embeddings. Re-export with `ExportOptions.LlamaCppContextLength = 128` (position interpolation), serve with `llama-server -c 128`, and set per-reply length with `max_tokens`/`n_predict`.

**"A GGUF fails to load: tensor layout does not match this build's quantization types."**
The file uses quant types newer than the installed llama.cpp build (publisher-fork dynamic quants, MXFP4). Diagnose with `GgufDiagnostics.Inspect(path)`, pick another quant, or re-quantize an F32 source with `llama-quantize`.

**"Export throws UnauthorizedAccessException / access denied."**
You pointed the exporter at a directory it cannot write. Export APIs accept a file path (an existing directory target writes `<dir>/<name>.gguf` inside it); pick a writable location.

**"The model has no search / thinking / tools inside llama.cpp."**
Those are runtime modules, not weights. Metadata + chat template declare them (so UIs gray them out); execution needs the Cognify runtime or an agent harness around the server (§9).

**"My training run's loss explodes / is NaN."**
Check `Decay` — it multiplies the LR every epoch. `Decay = 90` means LR × 90 per epoch. Use `0.9`–`0.95`. Also lower `LearningRate` (0.1 is very hot for a 32-dim model; 0.03 is plenty).

**"Early stopping fires immediately."**
`MinDelta` is in nats/token: `1e-2` is enormous for a tiny model whose loss improves by less than that per epoch late in training. Use `1e-4`.

---

## 12. Project structure, tests, license

```
Cognify/                       CognifyPy/
├── Cognify.sln                ├── cognify/
├── src/Cognify/               │   ├── core/         mind, builder, options, stimulus, events
│   ├── Core/                  # kernel, builder, cycle     ├── capabilities/ vision, audio, imaging, code, media,
│   ├── Reasoning/             # facts, rules, inference    │                  embeddings, multilingual, tools, web
│   ├── Thinking/              # chain of thought           ├── thinking/     chain of thought, thought generator
│   ├── Learning/              # MicroNet, DeepNet,         ├── learning/     micro_net, deep_net, adam, next_token,
│   │   └── Deep/              #   TransformerLM, Hybrid    │                 transformer, hybrid, learner, qlearn
│   ├── Emotions/              # appraisal, mood            ├── emotions/     appraisal, emotion_state, feeling, personality
│   ├── Memory/                # STM/LTM/episodic           ├── memory/       memory + associative network
│   ├── Goals/                 # drives, planner            ├── goals/        drives, goals
│   ├── Personality/           # Big Five                   ├── reasoning/    facts, knowledge, inference, arithmetic
│   ├── Perception/            # vision                     ├── compose/      response composer
│   ├── Audio/                 # hearing                    ├── persistence/  snapshot store
│   ├── Embeddings/            # hashed n-gram embedder     ├── export/       gguf, safetensors, tokenizer, llamacpp, diagnostics
│   ├── Multilingual/          # detection, phrasebook      └── demo.py / cli.py / __main__.py
│   ├── Tools/                 # registry, calculator
│   ├── Code/                  # generation, analysis       tests/            pytest suite (mirrors the xUnit set)
│   ├── Generation/            # media, WAV synthesis
│   ├── Export/                # GGUF/safetensors/llama.cpp
│   └── Persistence/           # SQLite store
├── demo/Cognify.Demo          # showcase
├── demo/Cognify.Yume          # Yume walkthrough
└── tests/Cognify.Tests/       # 518 xUnit tests
```

Both suites stay green on every change (C# `dotnet test` — 518 tests; Python `pytest`). CognifyPy is the behavioral mirror: features land on both sides with matching names, defaults and reports. Determinism is seeded (`Seed = 42`); the only C# dependency is `Microsoft.Data.Sqlite`; CognifyPy uses the standard library only. License: MIT.

