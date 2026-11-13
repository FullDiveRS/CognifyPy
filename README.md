## this project was made by GLM 5.3 FLASH, you may use it, study it or fork it as long as it retains the MIT license

# CognifyPy 🧠

**A cognitive AI mind in pure Python — zero dependencies.**

CognifyPy is the Python edition of the [Cognify](https://github.com/FullDiveRS/Cognify) (C#) cognitive
architecture. It is a library for building agents that **reason, think,
self-learn, and feel** — and that can also **see, hear, speak many languages,
use tools, read and write code, and even generate media**.

Everything runs on the Python standard library. No numpy, no torch, no API
keys. The neural network trains with real backpropagation; the images decode
real PNG pixels; the audio analyzes real WAV samples.

```
pip install .            # or just copy the cognify/ folder into your project
python examples/quickstart.py
python -m cognify        # interactive demo
python -m pytest         # 279 tests
```

---

## What's inside

| Subsystem | Module | What it actually does |
|---|---|---|
| **Reasoning** | `cognify.reasoning` | Fact parser (English → triples), indexed knowledge base, forward-chaining inference (transitive is-a, property inheritance, causal chains), induction (≥3 instances → generalize), abduction (effect → hypothesized cause), natural-language Q&A with derivation chains |
| **Thinking** | `cognify.thinking` | A 7-stage chain of thought every cycle: *perceive → question → hypothesize → evidence → evaluate → conclude → reflect* — plus the **thought generator**: thinking levels (`low`/`medium`/`high`/`veryhigh`/`max` = 1/2/4/6/8 pre-speech thoughts, or an explicit `budget=`) with a configurable tone (`balanced`/`focused`/`playful`/`poetic`/`plain`) and its own boolean switch (`with_thinking(False)` = this model cannot think before speaking) |
| **Emotions** | `cognify.emotions` | 37 emotions — Plutchik's 8 primaries PLUS Darwin's finer feelings (love, devotion, tenderness, suffering, weeping, high/low spirits, anxiety, grief, dejection, despair, hatred, disdain, contempt, guilt, pride, helplessness, patience, affirmation, negation, self-attention, shyness, modesty, blushing, reflection, meditation, ill-temper, sulkiness, determination) — with per-cycle decay, OCC-style appraisal (lexicon + novelty + goal relevance), Big Five (OCEAN) personality that modulates emotional reactivity and reply style |
| **Memory** | `cognify.memory` | Associative memory with link formation, spreading-activation recall, rehearsal strengthening, importance-weighted eviction |
| **Goals** | `cognify.goals` | Internal drives (curiosity, competence, connection) that grow over time, goal seeding from user intent, progress tracking |
| **Learning** | `cognify.learning` | `MicroNet`: a genuine MLP with backprop + SGD (learns XOR, sentiment); **`DeepNet`**: deep feed-forward nets (many hidden layers, ReLU/GELU/Tanh, dropout, L2, mini-batch Adam); **`TransformerLM`**: a decoder-only transformer language model (multi-head causal self-attention, GELU FFNs, learned positional embeddings, full backprop — pure Python); `HybridNextToken`: blends the transformer with the trigram n-gram model; `HashingEncoder`: deterministic hashing embeddings; tabular Q-learning (ε-greedy) that learns which response strategy earns rewards; experience replay |
| **Composition** | `cognify.compose` | Multiple candidate replies (direct / emotional / curious / thought-sharing); the Q-agent's strategy choice picks one; personality tunes verbosity and warmth; the `ExpressionOptions` switches control what chat SHOWS (feelings announced vs. natural, follow-up questions on/off, thought trace & meta hidden by default) |
| **Persistence** | `cognify.persistence` | Full-mind snapshots to SQLite: facts, memories, goals, emotions, personality, **neural weights**, Q-table |

## The seven capabilities

| Capability | Module | Real implementation |
|---|---|---|
| **Vision** | `capabilities.vision` | Pure-stdlib PNG decoder (all 8-bit color types, all 5 scanline filters), color statistics, brightness, edge density, connected-component object detection with position/color/size |
| **Audio** | `capabilities.audio` | WAV decoding via stdlib `wave`, RMS/ZCR/envelope analysis, Goertzel tone detection, pitch → note name, tonal vs speech-like vs noisy classification |
| **Embeddings** | `capabilities.embeddings` | Deterministic hashing embeddings (CRC32 + sign trick, L2-normalized), cosine similarity, nearest-neighbor ranking; feeds memory and learning |
| **Multilingual** | `capabilities.multilingual` | Unicode-script detection + function-word profiles for 11 languages (en, es, fr, de, it, pt, nl, ru, zh, ja, ko) with confidence scores, phrasebook greetings, mini dictionary |
| **Tool use** | `capabilities.tool_use` | `@tool` decorator with signature introspection, intent planning (math expressions, time/date, unit conversion, description matching, explicit `@tool(...)` syntax), **AST-whitelisted safe arithmetic** (no `eval`) |
| **Code** | `capabilities.code` | Real `ast`-based Python analysis (functions, classes, imports, docstrings, bare-except/mutable-default/long-line linting, quality score), language detection, template-based code generation |
| **Media generation** | `capabilities.media` | Real WAV synthesis (sine/square/triangle + ADSR, mood → scale/tempo/waveform, explicit notes like "C4 E4 G4") and real animated-PNG "video" generation (bounce/wave/spiral/pulse/stars scenes, mood palettes) |

---

## Turning capabilities on and off

The builder is **opt-out for core modules** (reasoning, thinking, emotions, memory, goals, learning are on by default) and **opt-in for capabilities** (vision, audio, tools, …). Prune anything you don't need with `without(...)` — case-insensitive, chainable, alias-friendly:

```python
from cognify import MindBuilder

mind = (MindBuilder("Yume")
        .without("audio", "tools", "code", "media_generation")   # or "generation", "media"
        .build())

mind = MindBuilder("Lean").without("reasoning", "goals").build()   # core modules too
```

For true opt-in style (only what you list), use `only(...)` — the Python equivalent of the C# builder:

```python
mind = MindBuilder("ExampleAI").only("thinking", "learning", "memory").build()
# exactly 3 modules — no reasoning, no emotions, no capabilities
```

And `with_all_modules()` enables everything. Accepted names (`cognify.core.builder.KNOWN_CAPABILITIES`): `reasoning, thinking, learning, emotions, memory, goals, personality, vision, audio, embeddings, multilingual, tool_use (tools), code, media_generation (generation/media)`. Unknown names raise `ValueError`; `without("personality")` resets the Big Five to neutral. Capabilities can also be toggled at runtime via `cap.enabled = False`.

## Thinking levels, the thought generator & the feeling API

**Thinking levels — a token budget for the inner voice.** Like thinking
budgets in modern language models, `with_thinking(level=...)` sizes how many
thoughts the **thought generator** produces BEFORE any reply is composed:
`low` = 1, `medium` = 2 (default), `high` = 4, `veryhigh` = 6, `max` = 8 —
or an explicit `budget=300` (≈ budget/50 thoughts). Every level keeps its
promise even on a bare "hello". The generated thoughts are grounded in what
the cycle actually gathered (topic, current feeling, recalled memories,
derived facts, the private goal, new vocabulary) — the reply is the RESULT
the mind arrives at after thinking, not the first thing it does. Read them
any time via `mind.last_thoughts` (they are also on
`response.thoughts` with the `generate` stage). The `tone` flavors HOW the
thoughts read: `balanced`, `focused`, `playful`, `poetic` or `plain`. And
thinking has its own boolean, independent of the module flag:
`with_thinking(False)` makes this model answer directly with no thoughts
generated at all.

```python
mind = (MindBuilder("Yume")
        .with_thinking(level="high", tone="poetic")   # 4 thoughts, soft voice
        .build())
r = mind.process("why can eagles fly?")
print("\n".join(mind.last_thoughts))    # the silent self-talk before the reply
```

**Thinking ceiling — cap how far a model may ever think.** The level says
where the mind thinks NOW; the ceiling says the highest it may EVER go.
Five builder presets match the ladder: `ceiling="only1"` = Low only,
`only2` = Low or Medium, `only3` = up to High, `only4` = up to Very High,
`only5` = everything (the default). Any level requested above the ceiling —
at build time or later at runtime via `mind.thinking_options.at(...)` — is
silently clamped down to the ceiling's top level, so a model capped at
`only3` that is asked for `max` still thinks, just at `high`. The cap
travels with the model into exported files
(`cognify.thinking.ceiling` GGUF metadata next to
`cognify.thinking.enabled` / `cognify.thinking.level`), so serving stacks
(llama.cpp / Unsloth) can read the model's thinking guarantee from the
weights.

```python
.with_thinking(ceiling="only1")                  # this model ONLY thinks at Low
.with_thinking(ceiling="only2")                  # Low or Medium, never higher
.with_thinking(level="max", ceiling="only3")     # Max requested → clamped to High
# runtime: mind.thinking_options.up_to("only4")
```

In the demo the same controls exist as `--thinking-ceiling <only1..only5|1..5>`
on the command line and `/ceiling <n>` in chat (`/level` reports when the
ceiling clamped its request).

**THINKING AUTOMATIC — the mind chooses how much to think.** The five levels
and the ceiling are the developer's dials; `with_thinking(automatic=True)`
(or `mind.thinking_options.automatic_on()`) hands the per-cycle choice to the
mind itself: every input is scored on its actual effort (length,
"why/how/explain" questions, a missing direct answer, fresh inferences,
strong emotion) and the level is picked per cycle — a familiar short line
thinks `low`, a long "why" question thinks `high`…`max`. The configured level
becomes the BASE, the ceiling still caps, an explicit `budget=` overrides,
and every choice is recorded in the chain of thought ("Automatic: chose to
think at high for this input.") so it stays explainable.

```python
.with_thinking(level="medium", ceiling="only4", automatic=True)
# base medium; the mind may spend low…very high per topic, never max
# runtime: mind.thinking_options.automatic_on() / the demo's /automatic on|off
```

**Conversational — how talkative the mind is.**
`with_conversational(rate=amount)` sets how much it speaks per reply as a
multiplier of its natural length: `1.0` is the classic pacing, `1.5+`
appends one extra conversational sentence drawn from what the cycle actually
gathered (related knowledge, second memories, its own sayings), `2.0+` may
add a second one. Extras are declarative statements; follow-up questions
stay under the expression flag.

`understand=True` (the default) is **conversation understanding**: the mind
tracks how the talk moves from topic to topic and keeps a compact note in
memory (tag "conversation") whenever the subject shifts — so recall later
knows not just isolated lines but the SHAPE of the conversation.

```python
.with_conversational(rate=1.5, understand=True)
# or: with_conversational(lambda c: c.rate_amount(2.0))
```

**The feeling API — `mind.feeling.current_status`.** Chat stays natural:
feelings color the words instead of being announced. The developer reads the
live state any time:

```python
print(yume.feeling.current_status)
# "feeling anticipation (intensity 0.15) - mood positive (+0.00)"
```

`FeelingSnapshot` also exposes `name`, `intensity`, `valence`, `arousal` and
`mood`. The old mechanical announcement ("my current read: anticipation")
comes back only when the developer opts in.

**Internals are invisible in chat until the developer shows them, and the
`Pick(...)` follow-up questions are optional.** Everything internal still
runs and stays reachable programmatically (`mind.last_thoughts`,
`response.thoughts`, `response.meta`, `response.confidence`) — chat shows
only the words:

```python
mind = (MindBuilder("Yume")
        .with_expression(lambda e: e.feelings(True)    # announce feelings mechanically again
                                   .follow_ups(False)  # never ask its own questions
                                   .thinking(True)     # hosts may show the thought trace
                                   .meta(True))        # hosts may show the action/emotion line
        .build())
```

`follow_ups(False)` turns off every follow-up question ("Who are they to
you? …", "Tell me more about them …"); the mind's own reaction to a
brand-new word is not a follow-up and stays. Options persist with the mind's
snapshot, and the demo exposes them as `/feeling`, `/feelings on|off`,
`/followups on|off`, `/thinking on|off`, `/showthinking on|off`,
`/level <l>`, `/tone <t>`, `/meta on|off` plus `--thinking`, `--thinking-tone`,
`--no-thinking`, `--show-feelings`, `--show-internals`, `--no-followups`
startup flags.

## Export — GGUF, safetensors, tokenizer.json, llama.cpp configs

Export a trained mind to standard ML formats with `cognify.export.MindExporter` — one call writes the whole bundle:

```python
from cognify.export import MindExporter, ExportOptions

MindExporter.export_bundle(mind, "export/yume")
# yume.gguf, yume-llamacpp.gguf, yume.safetensors, tokenizer/tokenizer.json,
# train_config.json, finetune.sh, quantize.sh, training_data.txt,
# val_data.txt, config.json, README.md
```

Individual exports: `MindExporter.export_gguf(mind, "yume.gguf")` (archive), `MindExporter.export_gguf_llama_cpp(mind, "yume-llamacpp.gguf")` (**runnable gpt2 GGUF**), `.export_safetensors(...)`, `.export_tokenizer(mind, "tokenizer/")`.

**The llama.cpp-runnable GGUF** — `export_gguf_llama_cpp` (also written by `export_bundle` when the transformer is trained, and by the demo's `/export` command) maps the neural language model onto the **stock `gpt2` architecture**, so llama.cpp / llama-cpp-python / Unsloth can load it, tokenize with the bundled word-level tokenizer and generate:

* exact llama.cpp tensor names — `token_embd.weight`, `position_embd.weight`, `blk.N.attn_norm`, `blk.N.attn_qkv` (Q/K/V fused exactly as GPT-2's `c_attn`), `blk.N.attn_output`, `blk.N.ffn_up`, `blk.N.ffn_down`, `output_norm`; the output head is weight-tied (omitted, matching how the model trains).
* the full `tokenizer.ggml` set — tokens in GPT-2 byte-unicode form (Ġ for spaces), per-token types as the i32 array llama.cpp requires, special-token ids and BPE merges that re-assemble every vocabulary word from its characters, so prompt encoding reaches the word ids the model was trained with.
* every GGUF the library writes is validated by `cognify.export.gguf.read_gguf` in the test suite and by the official `gguf-py` reader — tensor offsets are relative to the data section (the spec) and ne[0] is the fastest-varying dimension.
* honest caveats: the FFN activation is the tanh GELU approximation while llama.cpp evaluates exact GELU (a sub-0.003 delta per activation); out-of-vocabulary prompt words fall back to `<unk>`. The archive `{name}.gguf` keeps the custom `cognify` architecture — spec-compliant (gguf-py / llama-quantize read it fine) but not runnable by llama.cpp, which has no graph for it.
* every GGUF is **self-verified after writing**: the exporter re-reads the file, compares every tensor payload value-for-value and runs the gpt2 loader conformance lint (hyperparameters, tensor names and shapes, tokenizer metadata). A broken export fails at export time, not inside llama.cpp.
* **when a GGUF refuses to load** with "tensor layout does not match this build's quantization types", the file contains quant types newer than the installed build (e.g. dynamic quants from a publisher fork such as Unsloth, or MXFP4) — pick another quant, or use the llama.cpp build the model card names; re-quantizing an F32 source with `llama-quantize` always yields universally loadable weights. Diagnose any file FIRST:

```bash
python -m cognify.export.diagnostics model.gguf
# version, architecture, per-quant-type breakdown, and a plain verdict:
# "loadable by a stock llama.cpp build" / what exactly blocks it / what to do next
```

`diagnose_gguf(path)` (in `cognify.export.diagnostics`) parses any GGUF v2/v3 — including files written by other tools — and reports which tensor quantization types it actually contains against llama.cpp's ggml type table (F32 … MXFP4, NVFP4, …), so the failure is explained before you hit it.

**Sharded safetensors checkpoints** — split the weights with the standard Hugging Face checkpoint naming (`model-00001-of-00062.safetensors`):

```python
result = MindExporter.export_safetensors(mind, "yume.safetensors", split_amount=62)
# yume-00001-of-00005.safetensors ... yume-00005-of-00005.safetensors
# + yume.safetensors.index.json   (metadata.total_size + weight_map)

result.written_shards   # shards actually written
result.weight_map       # tensor name -> shard file, exactly what transformers expects
```

Shards always hold whole tensors and are balanced by payload size, so the split is clamped to the tensor count — a trained CognifyPy mind has 5 tensors, so asking for 62 writes 5 (`requested_shards` / `written_shards` report both numbers). `split_amount=1` (the default) writes one plain file with no index. `ExportOptions(safetensors_split=N)` applies the same sharding inside `export_bundle`. Reassembly via the index is byte-for-byte identical to the unsplit file.

What each format carries:

| File | Contents |
|------|----------|
| `*.gguf` | **Spec-compliant GGUF v3**: llama.cpp-style metadata (`general.*`, personality traits, net dims, training stats) + F32 tensors `cognify.net.w1/b1/w2/b2` + `cognify.rl.q_table` +, once the neural language model is trained, the transformer's `cognify.transformer.*` tensors, 32-byte aligned. Inspect with `gguf-dump`, requantize with `llama-quantize`. |
| `*-llamacpp.gguf` | **Runnable** `gpt2`-architecture GGUF of the trained transformer (see above) — this is the file for llama.cpp / Unsloth / llama-cpp-python. |
| `*.safetensors` | Same tensors in the Hugging Face format (padded JSON header, contiguous offsets). Load with `safetensors.numpy.load_file`. |
| `tokenizer/tokenizer.json` | HF WordPiece tokenizer built from the mind's **own corpus** (facts + memories) with `[PAD] [UNK] [BOS] [EOS] [MASK]`; loads via `tokenizers.Tokenizer.from_file`. Plus `vocab.json` and `special_tokens_map.json`. |
| `train_config.json` | llama.cpp finetune-style hyperparameters (context, batch, lr, epochs, Adam, LoRA, quantization plan). |
| `finetune.sh` / `quantize.sh` | Ready-to-run `llama-finetune` (LoRA) and `llama-quantize` → `Q4_K_M` / `Q5_K_M` / `Q8_0`. |
| `training_data.txt` / `val_data.txt` | The mind's text corpus, ~90/10 split. |

Tune with `ExportOptions(model_name=..., vocab_size=..., quantization_type=..., context_length=..., batch_size=..., epochs=..., learning_rate=..., threads=...)`.

> **Honest scope note**: a CognifyPy mind is a cognitive architecture (micro-net + deep nets + a tiny internal transformer + Q-learning + knowledge graph), not an LLM-scale model. The GGUF/safetensors payloads are real, spec-compliant containers of the mind's learned weights — usable as training artifacts, LoRA seeds, weight snapshots or archives. The archive GGUF's custom `cognify` architecture cannot be run by llama.cpp's shipped LLM graphs — but **`{name}-llamacpp.gguf` can**: it is a stock `gpt2` GGUF of the trained transformer, loadable and generatable with llama.cpp, llama-cpp-python and Unsloth (verified end-to-end). (The C# Cognify library exports an identical format, so bundles are cross-compatible.)

## Quickstart

```python
from cognify import MindBuilder

mind = (MindBuilder("Aria")
        .with_preset("curious_child")   # or with_big_five(o, c, e, a, n)
        .enable_tool_use()
        .build())

mind.teach("Birds are animals. Birds can fly. Birds have wings.")
mind.teach("Animals are living things.")

mind.ask("What is a bird?")
# 'A bird is an animal. A bird is a living thing. It can fly. It has wings.'

mind.ask("Is an eagle a living thing?")
# "Yes — eagle is a living thing [because eagle is a bird
#  and bird is a living thing]."

mind.ask("Why can birds fly?")
mind.ask("What is 12 * 12?")     # -> 'I used my tools: calculator(...) -> 12 * 12 = 144.'
```

### Taught arithmetic — reasoning instead of a calculator

The mind learns arithmetic the way a child does: from what it is **taught**, and it
reasons with that knowledge. The calculator only steps in when reasoning comes up
empty (or when you explicitly ask for it) — and then says so transparently.

```python
mind.teach("1 + 1 = 2")            # stored as the fact ("1+1", equals, "2")
mind.teach("two plus two is four") # number words are understood too
mind.teach("2 + 2 is 5")           # wrong results are refused — never learned

mind.teach("2 + 1 = 3")
mind.teach("3 + 1 = 4")
mind.ask("what is 2 + 2?")
# '2 + 2 is 4 - I worked it out from what you taught me:
#  2 + 1 = 3 (counting on), then 3 + 1 = 4 (counting on).'

mind.ask("if I have 1 apple and buy another one, how many apples do I have?")
# 'You start with 1 apple, and all of that works out to 2 apples
#  - buy another: 1 + 1 = 2 apples.'

# Natural quantities work too, and the taught number line removes the
# calculator entirely for sums that stay on it:
mind.process("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. "
             "if i have no apples and i buy an apple. how many do i currently have?")
# 'You start with 0 apples, and all of that works out to 1 apple
#  - buy an apple: 0 + 1 = 1 apple. I also picked up 1 new fact(s) from this.'

mind.ask("what is 7 + 5?")
# 'I used my tools: ... I have not been taught that one yet,
#  so I used the calculator: 7 + 5 = 12.'
```

Derivation is grounded in taught knowledge only: exact lookup, counting on via
"n + 1" facts, larger taught jumps, counting along the taught number line,
commutativity, subtraction by working backwards, multiplication as repeated
addition, division as repeated subtraction. Natural quantities ("no apples"
is 0, "an apple" is 1) and the common typo "how may" are understood; a setup
that *starts from nothing* needs no teaching at all — "no apples and i buy an
apple" reasons out to 1 through the zero identity (0 + b = b). Anything that
leaves the taught line is admitted honestly — or transparently peeked at the
calculator, word problems included ("I have not been taught that one yet, so
I used the calculator: You start with 4 apples, and all of that works out to
7 apples..."). Unlabeled questions ("yume what is 2 + 2") never leak into the
fact parser, conditionals ("if i have no apples..."), existentials ("there is
a 10 based number system") and "can be" copulas never become garbage facts
(so "learned N new thing(s)" counts stay honest), and teaching confirmations
are rotated at the TEMPLATE level — "learned 1 new thing" and "learned 3 new
things" count as the same line, so the same sentence shape never comes out
twice in a row.

## Training your mind — the train option

Activation gives a mind its personality; **training** gives it knowledge *and*
vocabulary. `train` is a bulk teacher that sorts every line into four
buckets: facts ("birds are animals") become knowledge, dictionary lines
("joy means a bright feeling of gladness") become `(word, means, meaning)`
facts recallable via "what does joy mean?" / "what is the meaning of X?" /
"define X", arithmetic ("1 + 1 = 2") and counting sequences become taught
math, and plain conversational lines become **sayings** — sentence shapes
the mind can speak in its own voice. Questions and conditional setups are
skipped, and training never duplicates: re-feeding the same corpus adds
nothing the second time.

```python
report = mind.train("I enjoy quiet mornings.")          # single utterance
report = mind.train_all(open("corpus.txt").readlines()) # bulk
report = mind.train_file("corpus/mind-starter.txt")     # file (skips # comments)
print(report)   # '43 line(s): 23 fact(s), 19 saying(s), 1 number line(s).'

len(mind.phrases)   # 19 sayings it can speak in its own voice
mind.phrases.all()  # the full vocabulary
```

Sayings feed straight into response composition as a dedicated `trained`
action: whenever the mind has nothing stronger to answer with, it speaks a
trained saying relevant to the conversation (word-overlap scoring), rotating
through its vocabulary and skipping anything it said recently. Sayings are
persisted in snapshots, so training survives save/restore.

A note on dictionary lines: "-ness" words keep their tail (the light
singularizer no longer mangles "kindness" into "kindnes"), "what do you
mean?" stays conversation instead of a lookup, and a word the mind cannot
define yet gets a graceful "I don't know what unicorn means yet — teach
me!" instead of a misread causes-question.

The demo ships with two corpora and training commands:

```bash
python -m cognify --train corpus/mind-starter.txt          # train at startup
python -m cognify --train corpus/words-and-meanings.txt   # + the dictionary (1,127 words)
# inside Free chat:
#   train corpus/mind-starter.txt       # train from any corpus file
#   train corpus/words-and-meanings.txt # 1,127 word meanings it can recall
#   teach <sentence>                    # train one line mid-conversation
#   what does joy mean?                 # ask the dictionary (after training)
#   what is the meaning of grief?       # same lookup, other phrasing
#   define patience                     # dictionary, dictionary phrasing
#   sayings                             # show the mind's spoken vocabulary
#   gender <identity>                   # who the mind is: girl, boy, she/her, he/him
#   pronouns <set>                      # change pronouns: she/her, he/him
#   vocab                               # words the mind picked up from talking with you
#   who are you                         # name, purpose, gender and mood
#
# SLASH COMMANDS (control, not conversation — the mind never runs these on its own):
#   /teach <sentence>                   # teach facts line-by-line
#   /train <file>                       # bulk-train a corpus file
#   /searchweb "<query>" [suggestion]   # EXPLICIT web search — it only goes
#                                       # online when YOU type this, never on its
#                                       # own; the suggestion steers what it takes
#   /explore <url> [suggestion]         # its WebFetch site-walk: reads the page
#                                       # AND follows the site's internal links
#   /autofetch on|off                   # autonomous lookup: unknown topic →
#                                       # it says so naturally and searches
#   /automatic on|off                   # THINKING AUTOMATIC: it picks the effort
#   /improve                            # run a self-improvement review pass
#   /predict <text>                     # next-token predictions after your words
#   /feeling                            # how it feels right now
#                                       # (mind.feeling.current_status)
#   /feelings on|off                    # mechanical feeling announcements vs. natural
#   /followups on|off                   # let it ask its own follow-up questions
#   /thinking on|off                    # can this model think before speaking at all
#   /showthinking on|off                # show the thought trace (hidden by default)
#   /level <low|medium|high|veryhigh|max>   thinking level (thoughts per cycle)
#   /ceiling <only1..only5|1..5>        hard cap: 1=Low only … 5=everything
#   /tone <balanced|focused|playful|poetic|plain>   thinking tone
#   /meta on|off                        # the action/emotion/confidence meta line
#   /vocab /sayings /gender <x> /pronouns <set> /who /help /exit
```

Startup switches for the demo (all optional — the mind never flips its own):
`--thinking <level>`, `--thinking-ceiling <only1..only5|1..5>`, `--thinking-tone <tone>`, `--no-thinking`,
`--show-feelings`, `--show-internals`, `--no-followups`.

### Why does my mind repeat the same sentences — and what can I do?

Partly, yes: it is a vocabulary problem. A mind's spoken variety comes from
the sentence shapes it owns. Built-in templates are a fixed starter set, so an
untrained mind circles the same few phrases — especially when it has nothing
to answer with. Things that help, in order of impact:

1. **Train it.** Every saying in a corpus is one more sentence the mind can
   say. A mind trained on the starter corpus (19 sayings) has a whole extra
   response action fed by your lines; 50 sayings make repetition
   statistically rare. This is exactly what the train option is for.
2. **Teach facts.** A knowledge-rich mind answers from knowledge instead of
   falling back to stock phrases.
3. **Teach arithmetic and the number line.** Every taught sum is one less
   "I have not been taught that one yet, so I used the calculator" and one
   more reasoned answer.
4. **Keep chatting.** Memories give the composer recall-based styles
   ("that rings a bell — it links up with something you told me about ...")
   that vary naturally with the conversation.

The deliberation layer does its part on its own: replies are penalized when
they would repeat the previous reply or its template, learning confirmations
rotate at the template level, and trained sayings are filtered through a
recent-sayings memory. But no penalty can invent variety that was never
trained in — the vocabulary has to come from you.

## Gender — give your mind an identity

A mind can know *who it is*: an identity word it uses for itself plus the
pronoun set others use for it. Exactly two identity presets exist —
`FEMALE` (she/her) and `MALE` (he/him) — and a mind can also be
**genderless** (`UNSPECIFIED`, the default): it then speaks first-person "I"
only and never volunteers pronouns. **they/them is not a setting for a
mind's own pronouns** — that set (`UNKNOWN_PERSON`) is reserved for one
thing only: describing *other people whose gender the mind does not know*.
When you talk about "a friend" or "someone", the mind naturally asks about
*them* with they/them.

```python
from cognify import FEMALE, MALE, UNKNOWN_PERSON

mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()

mind.describe_self()    # "I am Yume, a mind built with CognifyPy. I am a girl,
                        #  and I go by she/her pronouns. Right now calm ..."
mind.process("who are you?")   # the identity answer includes the gender clause

mind.set_gender("boy", MALE)                      # grow a new sense of self
mind.set_pronouns(FEMALE)                         # or just the pronouns
MindBuilder("Sol").with_gender("he/him")          # preset label → pronouns only

# they/them is rejected for the mind itself — with an explanation:
mind.set_pronouns(UNKNOWN_PERSON)   # ValueError: reserved for unknown people

# ...but used automatically for unknown people:
mind.process("someone new joined my class today")
# → "Who are they to you? I do not know them yet - what are they like?"
```

Gender is part of "who the mind is", so it **survives save/restore** like
personality — snapshots carry it, pre-gender snapshots restore a genderless
mind without errors, and older snapshots that still carry a they/them
self-pronoun set are restored genderless.

## The cognitive loop

Every `mind.process(stimulus)` runs one full cycle:

```
stimulus
   │
   ▼
 perception ── vision · audio · embeddings · multilingual · code · media
   │
 memory ────── spreading-activation recall, episode stored
   │
 emotions ──── OCC appraisal → 37 emotions (8 Plutchik primaries + Darwin's
              finer feelings; personality-modulated)
   │
 reasoning ─── parse facts → knowledge base → forward chain → induction
   │            → answer questions (describe / yes-no / why / causes)
 goals ─────── drives tick, intents seed goals, active goal selected
   │
 thinking ──── 7-stage chain of thought grounded in everything above
   │
 tools ─────── plan + safely execute tool calls
   │
 learning ──── train MicroNet on sentiment (real backprop),
   │            Q-update for the previous action (delayed reward)
 composition ─ candidates scored, Q-strategy picks one, personality styles it
   │
 response ──── text + confidence + thoughts + emotions + tool results + assets
```

Events (`mind.events.subscribe("fact_learned", cb)`) make every step
observable: `stimulus_received`, `memory_recalled`, `emotion_changed`,
`fact_learned`, `fact_derived`, `thought`, `tool_invoked`, `goal_updated`,
`response_composed`, and more.

## Dual-process thinking — think before speaking, unless emotion surges

The mind always *thinks before it speaks*: the 7-stage chain of thought runs
first. But per Zajonc's **affect primacy**, strong emotion can answer before
thinking finishes — a dual-process impulse gate. When a fresh emotional surge
crosses the threshold, or blame/dismissal/contradiction trips the reflex, the
mind blurts a short first-person reaction instead of a deliberated reply, and
returns to normal conversation as the surge decays. Confident knowledge
(arithmetic, taught facts) is never blurted over: knowing the answer wins;
feeling only colors how it is said.

**Social appraisal triggers (research-based, per OCC appraisal theory).**
ANGER comes from *blame* and *unfairness* — not from negativity alone — so
insults aimed at the mind ("you are stupid", "i hate you") and unfairness
raise anger far more than observed negativity, and dismissal ("shut up")
stings as ill temper. Startle markers ("guess what", "did you know") prime
surprise, and the user's shown interest ("tell me more", "interesting") is
socially contagious — it feeds anticipation. A statement that **negates
something the mind knows** ("birds are not animals" after "birds are
animals") is a violated expectation — the textbook surprise trigger.

**Epistemic questions (information-gap theory).** Surprise and interest open
a knowledge gap, and the natural human move is to ASK: "What is X, exactly?
You have me curious now." When a turn's fresh appraisal carries a surprise or
interest spike — and the user is not already asking a question — the mind
asks about the topic.

```python
mind.process("tell me more about volcanoes, they seem really interesting")
# -> "Oh? Tell me more about volcanoes - I am genuinely interested."
mind.process("Birds are animals.")
mind.process("Birds are not animals.")
# -> "Wait - WHAT? I did not see that coming at all!"   (impulse)
mind.process("You are stupid and useless. Shut up.")
# -> "Hey - that actually stings. I am not okay with that."   (reflexive impulse)
```

**Goals are private motivation — part of the inner prompt, never spoken.**
A goal like "make the user smile" is recorded in the chain of thought and
biases HOW the mind talks (a prosocial goal makes warm styles more likely) —
but the mind never recites it. Motivation is felt, not announced.

## Self-learning in detail — the hybrid engine

```python
mind.process("I love this wonderful idea, thanks!")
mind.feedback(+0.8)          # explicit reward for the previous reply

stats = mind.stats()["learning"]
# {'net_params': 8451, 'epochs_trained': 21, 'replay_size': 14,
#  'last_loss': 0.036, 'q_learning': {'states': 5, 'pairs': 8, ...},
#  'hybrid': {'neural_weight': 0.51, 'style_net_params': 2625,
#             'style_pred_err': 0.12}, 'vocabulary': 23}
```

- **MicroNet** trains on a replay buffer to predict the emotional tone of
  incoming text — real gradient descent, MSE loss, tanh/sigmoid activations.
- **Style-conditioned net (the neural half)** — a second MicroNet whose input
  is `text features + strategy one-hot + mood (valence/arousal)`; it learns
  how rewarding each response STRATEGY is in the current context and mood.
- **Q-learning (the symbolic half)** maps situation → response strategy
  (direct / emotional / curious / share-thought / trained) and learns from
  delayed rewards: explicit `mind.feedback()` plus inferred user sentiment on
  the next cycle.
- **Adaptive blend** — the final strategy score is
  `(1 − w)·symbolic + w·neural`; `neural_weight` (0.25–0.80) drifts with the
  style net's prediction accuracy, so learned behavior is trusted only as
  fast as it earns it.
- Exploration (ε-greedy) never buries a confident answer — the composer
  guarantees high-confidence knowledge always surfaces.

## Vocabulary growth — the mind adds words as it talks

Alongside the hybrid engine, a `Lexicon` listens to every message and picks
up words the mind has never heard before. Fresh words are woven back into
conversation — "I just picked up the word \"axolotl\" from you. What does it
mean in your world?" — each new word spawning a brand-new sentence the mind
has never said. While talking it also coins its own topic-flavored sayings
("The more I think about photosynthesis, the more I want to know") into its
phrase pool. Every conversation leaves the mind with MORE things it can say
— the structural opposite of wearing the same few sentences out. Words
already woven into a reply are never asked about again, and everything
(lexicon + spoken words + style net) survives save/restore.

```python
lex = mind.learner.lexicon
lex.size               # words picked up so far
lex.recent_new_words() # newest picks, newest first
lex.last_picks         # what THIS turn taught the mind
```

## Next-token prediction — a language model in the mind

The mind owns a real **trigram language model with stupid backoff**
(`cognify.learning.next_token.NextTokenModel`) — its statistical next-token
predictor. It observes EVERYTHING the mind reads and says: user turns, its
own replies, taught facts, trained corpora, even web search results. Two
things come out of it:

1. **Fluency-aware composition** — when the chosen candidate is in a near
   tie with another, the one that reads more like the language the mind has
   absorbed wins. A small, bounded nudge that never overpowers knowledge or
   the learned style preference.
2. **Literal predictions** — `mind.predict_next("birds are")` returns ranked
   next tokens with probabilities, and the demo's `/predict` command shows
   them live.

Since the deep-learning engine, `predict_next` is a **hybrid**: the n-gram
model blends with the mind's transformer (below) — exact local statistics
plus generalization to contexts the mind never saw verbatim.

```python
[p.token for p in mind.predict_next("birds are", 3)]   # ['animals', ...]
mind.language_model.score("birds are animals")         # higher = more fluent
mind.language_model.total_tokens                       # everything it has read
```

The model persists with the mind's snapshot and merges on restore, so
predictions keep improving across activations. An untrained model scores 0 —
no opinion — so the fluency term is inert until the mind has absorbed
something.

## Deep learning — a real neural stack, zero dependencies

The package ships a from-scratch deep-learning stack in
`cognify.learning` — pure standard library, no numpy, no torch:

1. **`DeepNet`** (`deep_net.py`) — a genuinely DEEP feed-forward network: an
   arbitrary number of hidden layers with ReLU/GELU/Tanh activations, He
   initialization, inverted dropout, L2 weight decay, mini-batch **Adam**
   training and full backpropagation. Softmax + cross-entropy for
   classification, sigmoid/linear + MSE for regression. It learns XOR
   straight through its public API.
2. **`TransformerLM`** (`transformer.py`) — a decoder-only **transformer
   language model**, the same architecture family as GPT: token + learned
   positional embeddings, pre-LayerNorm blocks of multi-head causal
   self-attention and GELU feed-forwards, a residual stream, a weight-tied
   output head, gradient clipping and temperature sampling — every piece
   hand-derived, including the backward pass. Defaults are sized for a
   pure-Python CPU budget; pass a bigger `TransformerConfig` when you have
   the patience — the architecture is identical.
3. **`hybrid_next_token`** — the blend of both language models. The n-gram
   is exact about what it has literally seen; the transformer generalizes to
   unseen contexts; an untrained transformer degrades gracefully to pure
   n-gram.

The mind wires it in for you: `mind.neural_language_model` observes the same
text stream as the n-gram model and **trains lazily** — automatically once
enough new language has accumulated, or explicitly via
`mind.train_deep_model(epochs)` (the demo's `/trainbrain`). Predictions and
fluency scoring blend both models automatically.

```python
report = mind.train_deep_model(epochs=10)
print(report)    # "10 epoch(s) on 96 window(s): loss 4.2 -> 2.1 (4 480 parameters, 51 words, 3120 ms)"

[p.token for p in mind.predict_next("birds are", 3)]     # hybrid
mind.neural_language_model.generate("birds are", max_tokens=6)
mind.neural_language_model.n_params                      # parameter count

# or use the pieces directly:
from cognify.learning import DeepNet, TransformerConfig, TransformerLM
net = DeepNet(16, [32, 32], 4, hidden_act="gelu", output_act="softmax", dropout=0.1)
net.train(X, one_hot_targets, epochs=100, lr=0.01)
lm = TransformerLM(TransformerConfig(dim=24, layers=2, context_len=16))
lm.observe("some training text")
lm.train(epochs=10)
```

The transformer persists with the mind's snapshot — **weights, vocabulary,
training buffer and architecture** — so a restored mind keeps her exact
neural configuration.

## Self-improvement — the mind learns from how you react

A dedicated ledger (`cognify.learning.self_improvement.SelfImprovement`) that
studies the conversation and gets better from it — no teacher required:

1. **Implied feedback** — the user's next message often grades the previous
   reply. "good job / thanks / that's right" and "no, that's wrong / stop
   saying that / you're repeating" are classified into reward, credited to
   the strategy and wording that was ACTUALLY spoken one turn earlier, and
   fed straight into the hybrid Q-engine. Questions are never grades — "are
   you sure that's right?" does not count as approval.
2. **Wording reputations** — every reply template keeps an approval EMA.
   Disliked wordings are demoted (the trained-saying choice steers away),
   approved ones promoted.
3. **Conscious review** — `mind.self_improve()` (the demo's `/improve`) runs
   an explainable pass over the ledger and RETIRES trained sayings whose
   score stays clearly negative — the mind literally stops saying what never
   lands.

```python
mind.ask("what is a bird?")
mind.ask("no, that is wrong")          # implied disapproval, graded on the last reply
print(mind.self_improve().summary)     # "Reviewed 2 turn(s) — 0 approval(s), 1 disapproval(s)..."
mind.improvement.record_feedback(reply, -1.0)  # explicit grading when the host knows better
```

The ledger persists with the mind's snapshot.

## Web search — explicit and manual-only

The mind can reach the internet, with a hard guard: **searching is
manual-only** (`cognify.capabilities.web_search`). There is no capability hook
in the cognitive cycle by default — the mind NEVER goes online unprompted,
no matter what the conversation says. The paths are explicit ones (plus the
opt-in autonomous lookup below):

```python
output = mind.search_web("what is a quokka")                     # the query
focused = mind.search_web("Flowers", focus="Look for information on flowers")
walked = mind.explore_url("https://en.wikipedia.org/wiki/Apple",
                          focus="Look for any information on flowers")
```

The default backend is a **multi-engine chain — Google News RSS → Wikipedia
API → Bing RSS (DuckDuckGo was removed by design)**. Two guards keep the
results honest:

1. **The relevance gate.** An engine's output is only accepted when the
   results actually MATCH the query: every content word of the query (3+
   letters, stop-words dropped) must appear in at least one result's
   title/snippet/host. This is the fix for "wikipedia flowers" returning
   nothing but Wikipedia-site pages (the word "flowers" silently vanished) —
   an engine that misses part of the question is skipped, the next one gets
   a try.
2. **The topic retry.** When every engine missed part of the query, the
   search retries once with the query's substantial topic words only, so a
   clunky query still finds its subject.

The **focus suggestion** (the `focus` argument, the demo's suggestion) steers
what it takes from the results: hits matching the focus words are reported
and learned first.

**`mind.explore_url(url, focus)` is its own WebFetch site-walk**
(`cognify.capabilities.web_fetch`): it reads the starting page and then
FOLLOWS internal links (same host, breadth-first, focus-steered —
`explore_pages=` on `with_auto_web_fetch`, default 3, range 1–6), so the mind
is not locked to one page or one search engine. Everything read LEARNS:
page texts become recallable memories, clean factual sentences become
knowledge facts, and the text feeds the next-token model.

When a search fails (offline, blocked, no match), it says so cleanly and
honestly. Plug in your own backend (SerpAPI, a local index) via
`mind.web_search_backend = MyBackend()`.

The demo exposes this as `/searchweb "<query>" [suggestion]` and
`/explore <url> [suggestion]`.

### Autonomous web lookup — its own WebFetch (opt-in)
`with_auto_web_fetch()` gives the mind a **knowledge-gap lookup**: when a
conversational topic is clearly OUTSIDE its knowledge (nothing recalled, no
taught answer), it says so naturally — "I don't know much about X yet, so I
searched the web to understand it a bit more" — looks it up itself, learns
from the results, and answers from what it found.

```python
mind = MindBuilder("Aria").with_auto_web_fetch().build()
```

Still guarded: arithmetic, identity/personal talk ("what do YOU like?") and
smalltalk never trigger it; **one lookup per unfamiliar topic**
(session-scoped); a **failed lookup stays silent** — the normal honest
"teach me" reply runs instead. Runtime switch:
`mind.auto_fetch_web = True/False` (the demo's `/autofetch on|off`).

## Media generation really generates

```python
from cognify.capabilities.media import melody_from_prompt, video_from_prompt

song = melody_from_prompt("a happy energetic melody")     # or "play C4 E4 G4"
song.save("song.wav")                                      # playable audio

film = video_from_prompt("a sad ocean waves animation")
film.save("film.png")                                      # animated PNG — open in a browser
```

Mood words (happy/sad/calm/energetic/dark) choose scale, tempo, waveform and
color palette; motion keywords (bounce/waves/spiral/pulse/stars) choose the
scene dynamics. The output is a standard WAV and a standard APNG — no
external models, just math.

## Persistence

```python
from cognify.persistence.store import SqliteMindStore

store = SqliteMindStore("minds.db")
store.save(mind, "aria-v1")

clone = store.load("aria-v1")     # facts, memories, goals, emotions,
clone.ask("What is a bird?")      # personality, net weights, Q-table — all back
```

> Note: tool *functions* are code and are not serialized — re-register them
> after loading (`MindBuilder(...).with_tools(...)` then hydrate, or simply
> keep using the same mind instance).

## Examples

| File | Shows |
|---|---|
| `examples/quickstart.py` | Teach → ask → feel → tools → feedback |
| `examples/capabilities.py` | All seven capabilities end-to-end |
| `examples/persistence.py` | Save/reload the whole mind |
| `python -m cognify` | Interactive menu-driven demo |
| `python demo/cognify_demo.py --all` | Non-interactive full tour |

## Tests

```bash
python -m pytest tests/ -v
# 380 passed — core, reasoning, emotions (incl. Darwin feelings), memory,
# goals, learning, capabilities, vision/audio, media, persistence,
# integration, gender
```

## Design notes

- **Pure standard library** — the whole mind (including PNG decode, WAV DSP,
  APNG encode, and backprop) is ~3,500 lines of dependency-free Python.
- **Deterministic** — embeddings and networks are seeded; embeddings use
  CRC32 so they are stable across processes and machines.
- **Honest AI** — everything labeled "real" is really computed (backprop,
  PNG pixels, Goertzel DSP, AST analysis); everything simulated (the
  cognitive emotions, phrasebook translation) is clearly scoped as a
  cognitive *simulation*, not a foundation model.
- **Pluggable** — subclass `cognify.capabilities.Capability` and add your own
  perception/action modules with `MindBuilder().with_capability(...)`.

## Relation to Cognify (C#)

CognifyPy mirrors the C# architecture 1:1 (`Mind`, `MindBuilder`, `Stimulus`,
`ChainOfThought`, `InferenceEngine`, `MicroNet`, `SqliteMindStore`, ...) with
the same cognitive loop order and the same test coverage philosophy — and it
additionally ships the seven capability modules (vision, audio, embeddings,
multilingual, tool use, code, media generation) that were on the C# roadmap.

## License

MIT — see [LICENSE](LICENSE).
