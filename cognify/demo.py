"""Interactive / scripted demo of the CognifyPy mind."""
from __future__ import annotations

import os
import sys

from .capabilities.imaging import encode_png
from .capabilities.media import melody_from_prompt, video_from_prompt
from .core.builder import MindBuilder
from .core.gender import FEMALE, MALE, UNSPECIFIED, PronounSet
from .core.stimulus import Stimulus


def banner(text: str) -> None:
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def make_full_mind() -> MindBuilder:
    return (MindBuilder("Aria")
            .with_preset("curious_child")
            .with_gender("girl", FEMALE)
            .with_thinking(level="high", tone="balanced",  # 4 thoughts before speaking
                           automatic=True)  # THINKING AUTOMATIC: it picks the
                                           # effort per topic (level = the base)
            .with_conversational(rate=1.5, understand=True)  # chattier replies +
                                            # it tracks how the talk shifts topic
            .with_auto_web_fetch()  # manual-only /searchweb + /explore PLUS
                                           # the autonomous lookup: unknown
                                           # topic → it says so, searches, learns
            .enable_embeddings()
            .enable_multilingual()
            .enable_tool_use()
            .enable_vision()
            .enable_audio()
            .enable_code()
            .enable_media_generation())


def _demo_reasoning(mind) -> None:
    banner("1. REASONING - teach facts, watch inference chain them")
    print("  taught:", mind.teach("Birds are animals. Birds can fly. Birds have wings."))
    print("  taught:", mind.teach("Animals are living things. Eagles are birds."))
    print("\nQ: What is a bird?\n  ->", mind.ask("What is a bird?"))
    print("\nQ: Can birds fly?\n  ->", mind.ask("Can birds fly?"))
    print("\nQ: Is an eagle a living thing? (2-hop inference)\n  ->",
          mind.ask("Is an eagle a living thing?"))
    print("\nQ: Why can birds fly?\n  ->", mind.ask("Why can birds fly?"))


def _demo_thinking(mind) -> None:
    banner("2. THINKING - levels (low..max), the thought generator, the 7-stage chain")
    opts = mind.thinking_options
    print(f"  this model can think: {opts.enabled} | level: {opts.level.value} "
          f"({opts.thought_rounds} thought(s) per cycle) | tone: '{opts.tone}'")
    response = mind.process("Why can eagles fly?")
    if mind.last_thoughts:
        print("\n  her thoughts BEFORE speaking (mind.last_thoughts):")
        for t in mind.last_thoughts:
            print(f"    ~ {t[:88]}")
    print("\n  the full chain (response.thoughts — internals, hidden in chat by default):")
    for stage, content, conf in response.thoughts:
        print(f"    [{stage:>12}] {content[:84]} ({conf:.2f})")
    print("\n  levels are a budget: low=1 / medium=2 / high=4 / veryhigh=6 / max=8 "
          "thoughts — or an explicit token budget (budget=300 -> 6).")


def _demo_emotions(mind) -> None:
    banner("3. EMOTIONS - praise it, scare it, watch it settle + the feeling API")
    for text in ("I love this, you are wonderful!",
                 "This is scary and dangerous!",
                 "hello"):
        r = mind.process(text)
        print(f"  '{text[:38]}' -> dominant={r.meta.get('dominant_emotion')} "
              f"valence={r.meta.get('valence', 0):+.2f}")
        print(f"    reply: {r.text[:78]}")
    # The feeling API: chat colors its words naturally, the raw state is
    # readable by the DEVELOPER any time (yume.feeling.current_status).
    print("\n  mind.feeling.current_status ->", mind.feeling.current_status)
    print("  (fields: name / intensity / valence / arousal / mood)")
    print("  chat announces feelings mechanically only when the developer sets "
          "feelings(True) — default is natural.")


def _demo_learning(mind) -> None:
    banner("4. SELF-LEARNING - hybrid engine (symbolic Q + neural style net)")
    for _ in range(3):
        mind.process("I love this wonderful idea, thanks!")
        mind.process("This is terrible and awful.")
    stats = mind.stats().get("learning", {})
    print("  net params:      ", stats.get("net_params"))
    print("  epochs trained:  ", stats.get("epochs_trained"))
    print("  replay samples:  ", stats.get("replay_size"))
    print("  last train loss: ", stats.get("last_loss"))
    print("  Q-learning:      ", stats.get("q_learning"))
    print("  hybrid blend:    ", stats.get("hybrid"))
    sentiment, prob = mind.learner.sentiment_of("what a brilliant amazing thing")
    print(f"  net classifies 'brilliant amazing thing' -> {sentiment} ({prob:.2f})")


def _demo_capabilities(mind, out_dir: str) -> None:
    banner("5. CAPABILITIES - vision, audio, multilingual, tools, code, media")
    os.makedirs(out_dir, exist_ok=True)

    # VISION - render a tiny scene (green field, red ball) and show it
    rows = [[(40, 180, 70)] * 48 for _ in range(36)]
    for y in range(10, 22):
        for x in range(18, 30):
            rows[y][x] = (220, 40, 40)
    r = mind.process(Stimulus.image(encode_png(48, 36, rows), text="What do you see?"))
    print("  VISION  ->", r.text[:95])
    print("           capabilities:", r.capabilities)

    # AUDIO - synthesize a melody, let the mind listen
    wav = melody_from_prompt("a calm melody").data
    r = mind.process(Stimulus.audio(wav, text="Listen to this."))
    print("  AUDIO   ->", r.text[:95])

    # MULTILINGUAL
    r = mind.process("Hola, gracias!")
    print("  LANG    ->", r.text[:95])

    # TOOLS
    r = mind.process("What is 12 * 12?")
    print("  TOOLS   ->", (r.tool_results or [r.text])[0][:95])

    # CODE analysis
    snippet = ("def double(x):\n"
               "    return x * 2\n"
               "class Thing:\n"
               "    pass\n")
    r = mind.process(Stimulus.code(snippet))
    print("  CODE    ->", r.text[:95])

    # CODE generation
    r = mind.process("Write a function that reverses a string")
    print("  CODEGEN ->", r.text[:95])

    # MEDIA generation - save a melody + an animation to disk
    a1 = melody_from_prompt("a happy melody", seed=3)
    a2 = video_from_prompt("a happy bouncing ball animation", seed=3)
    p1 = a1.save(os.path.join(out_dir, "melody.wav"))
    p2 = a2.save(os.path.join(out_dir, "animation.png"))
    print(f"  MEDIA   -> saved {p1} ({len(a1.data)} B WAV) and "
          f"{p2} ({len(a2.data)} B APNG)")


def _demo_memory_and_goals(mind) -> None:
    banner("6. MEMORY & GOALS - recall, drives, goal tracking")
    mind.process("I want to learn photography")
    mind.process("tell me more about photography please")
    st = mind.stats()
    print("  open goals:", st["goals_open"], "| facts:", st["facts"],
          "| memories:", st["memories"])
    goal = mind.goals.active
    if goal is not None:
        print(f"  active goal: '{goal.description}' ({int(goal.progress * 100)}%)")


def _demo_persistence(mind) -> None:
    banner("7. PERSISTENCE - save the whole mind to SQLite, reload it")
    from .persistence.store import SqliteMindStore
    store = SqliteMindStore("demo_minds.db")
    store.save(mind, "aria")
    clone = store.load("aria")
    print("  snapshots:", [s["name"] for s in store.list()])
    print("  clone knows:", clone.ask("What is a bird?"))
    print("  clone cycles:", clone.cycle_count, "| facts:", clone.kb.size())
    store.close()


def run_all(quiet: bool = False) -> None:
    """Run the complete tour non-interactively (used by CI and `--all`)."""
    if quiet:
        old = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            _run_all_inner()
        finally:
            sys.stdout.close()
            sys.stdout = old
    else:
        _run_all_inner()


def _run_all_inner() -> None:
    mind = make_full_mind().build()
    _demo_reasoning(mind)
    _demo_thinking(mind)
    _demo_emotions(mind)
    _demo_learning(mind)
    _demo_capabilities(mind, "cognify_media")
    _demo_memory_and_goals(mind)
    _demo_persistence(mind)
    banner("DONE - every subsystem exercised")


_MENU = {
    "1": ("Teach facts & ask questions", _demo_reasoning),
    "2": ("Thinking levels & the thought generator", _demo_thinking),
    "3": ("Play with emotions", _demo_emotions),
    "4": ("Learning statistics", _demo_learning),
    "5": ("Capabilities showcase", _demo_capabilities),
    "6": ("Memory & goals", _demo_memory_and_goals),
    "7": ("Save / load the mind", _demo_persistence),
    "8": ("Free chat", None),
}


def _split_query_and_focus(rest: str) -> "tuple[str, str]":
    """'/searchweb "Flowers" Look for any information on flowers' →
    ('Flowers', 'Look for any information on flowers'). Without quotes the
    whole rest is the query (backward compatible)."""
    rest = rest.strip()
    if len(rest) > 1 and rest[0] == '"':
        close = rest.find('"', 1)
        if close > 1:
            focus = rest[close + 1:].strip().lstrip("- :")
            return rest[1:close].strip(), focus
    return rest, ""


def _handle_slash(line: str, mind) -> bool:
    """Slash commands — the control channel. '/' keeps them strictly apart
    from conversation: nothing here is ever spoken by the mind on its own,
    and web search in particular ONLY runs when you type /searchweb yourself.
    Returns True when the chat should end (/exit)."""
    parts = line.split(" ", 1)
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/help", "/?"):
        print("[slash commands]")
        print("  /teach <sentence>      teach facts line-by-line")
        print("  /train <file>          bulk-train a corpus file")
        print('  /searchweb "<q>" [sugg]  web search — quoted part is the query, the rest steers what it takes from it')
        print("  /explore <url> [sugg]    read a page AND follow its internal links (its own WebFetch walk)")
        print("  /autofetch on|off        its autonomous lookup: unknown topic → it searches itself")
        print("  /automatic on|off        THINKING AUTOMATIC: it picks the effort per topic")
        print("  /improve               run a self-improvement review pass")
        print("  /predict <text>        show next-token predictions after your words")
        print("  /trainbrain [epochs]   deep-train the transformer language model")
        print("  /brain                 status of the transformer language model")
        print("  /export [dir]          export the model bundle (GGUF archive + llama.cpp-runnable gpt2 GGUF + safetensors + tokenizer)")
        print("  /feeling               how it feels right now (mind.feeling.current_status)")
        print("  /feelings on|off       announce feelings mechanically vs. keep chat natural")
        print("  /followups on|off      let it ask its own follow-up questions")
        print("  /thinking on|off       can this model think before speaking at all")
        print("  /showthinking on|off   show the thought trace in chat (hidden by default)")
        print("  /level <l>             thinking level: low|medium|high|veryhigh|max (clamped by the ceiling)")
        print("  /ceiling <n>           hard cap: only1..only5 — 1=Low only, 2=+Medium, 3=+High, 4=+VeryHigh, 5=everything")
        print("  /tone <t>              thinking tone: balanced|focused|playful|poetic|plain")
        print("  /meta on|off           show the response meta line (hidden by default)")
        print("  /vocab /sayings /gender <x> /pronouns <set> /who /exit")
        return False
    if cmd == "/teach" and rest:
        print(f"mind> {mind.train(rest)}")
        return False
    if cmd == "/teach":
        print("      [usage: /teach <sentence>]")
        return False
    if cmd == "/train" and rest:
        try:
            report = mind.train_file(rest)
            print(f"mind> trained: {report}")
            print(f"      [vocabulary: {mind.phrases.count} saying(s) it can speak now]")
        except FileNotFoundError as exc:
            print(f"      [training failed: {exc}]")
        return False
    if cmd == "/train":
        print("      [usage: /train <file>]")
        return False
    if cmd in ("/searchweb", "/search") and rest:
        # /searchweb "Flowers" Look for any information on flowers
        #   → query = Flowers (what to search), focus = the rest (which
        #     results matter most — it learns/reports those first).
        query, focus = _split_query_and_focus(rest)
        note = f' — focusing on: {focus}' if focus else ""
        print(f'[searching the web for "{query}"{note} — it only searches when you ask, never on its own]')
        output = mind.search_web(query, focus=focus or None)
        print(f"mind> {output}")
        if not output.startswith("I could not"):
            print("      [learned from the results: memories stored, clean facts added, next-token model updated]")
        return False
    if cmd in ("/searchweb", "/search"):
        print('      [usage: /searchweb "<what to look up>" [suggestion — what to focus on]]')
        return False
    if cmd == "/explore" and rest:
        parts2 = rest.split(" ", 1)
        url = parts2[0]
        focus = parts2[1].strip() if len(parts2) > 1 else ""
        note = f" — focusing on: {focus}" if focus else ""
        print(f"[exploring {url}{note}]")
        output = mind.explore_url(url, focus=focus or None)
        print(f"mind> {output}")
        if not output.startswith(("I could not", "No web url")):
            print("      [pages → memories (tag 'web'), clean sentences → facts, text → next-token model]")
        return False
    if cmd == "/explore":
        print("      [usage: /explore <url> [suggestion — what to look for inside the site]]")
        return False
    if cmd == "/autofetch":
        on = rest not in ("off", "false", "0")
        mind.auto_fetch_web = on
        print("[autonomous lookup ON — when a topic is clearly outside its knowledge it will say so and search itself]"
              if on else
              "[autonomous lookup OFF — it goes online only when you type /searchweb or /explore]")
        return False
    if cmd == "/automatic":
        on = rest not in ("off", "false", "0")
        mind.thinking_options.automatic_on(on)
        print(f"[THINKING AUTOMATIC {'ON' if on else 'OFF'}"
              + (f" — base {mind.thinking_options.level.value}, ceiling "
                 f"{mind.thinking_options.ceiling.value}" if on else "")
              + "]")
        return False
    if cmd == "/improve":
        report = mind.self_improve()
        print(f"mind> {report.summary}")
        return False
    if cmd == "/predict":
        context = rest or "I"
        predictions = mind.predict_next(context, 5)
        if not predictions:
            print("mind> My language model has not learned enough yet — train me and ask again.")
            return False
        rendered = ", ".join(f'"{p.token}" ({p.probability:.0%})' for p in predictions)
        print(f'mind> After "{context}" I would most likely say next: {rendered}.')
        nlm = mind.neural_language_model
        extra = (f" — transformer ({nlm.n_params} parameters, loss {nlm.last_loss:.3f}) is blending in too"
                 if nlm.is_trained else
                 " — transformer not trained yet: /trainbrain to train it")
        print(f"      [n-gram: {mind.language_model.total_tokens} token(s) observed, "
              f"{mind.language_model.vocabulary} word(s) known{extra}]")
        return False
    if cmd == "/trainbrain":
        epochs = 8
        if rest:
            try:
                epochs = max(1, min(100, int(rest.split()[0])))
            except ValueError:
                print("      [usage: /trainbrain [epochs 1-100]]")
                return False
        nlm = mind.neural_language_model
        print(f"      [deep training the transformer: {nlm.cfg.layers} layer(s), {nlm.cfg.dim}d, "
              f"{nlm.cfg.heads} heads, {nlm.n_params} parameters, "
              f"{nlm.buffered_sentences} sentence(s) buffered — {epochs} epoch(s)]")
        report = mind.train_deep_model(epochs)
        if report.epochs == 0:
            print("mind> I have not heard enough language yet — talk to me or /teach me first, "
                  "then train my brain.")
            return False
        print(f"mind> Done — {report}")
        print("      [my transformer now blends into /predict and how naturally I word my replies]")
        return False
    if cmd == "/export":
        from cognify.export.exporter import MindExporter
        out_dir = rest if rest else "export"
        print(f"      [exporting the model bundle to {out_dir}]")
        try:
            files = MindExporter.export_bundle(mind, out_dir)
            for path in files:
                print(f"      wrote {path}")
            if any(f.endswith("-llamacpp.gguf") for f in files):
                print("      [the -llamacpp.gguf file loads in llama.cpp / Unsloth / "
                      "llama-cpp-python — the transformer as a standard gpt2 model]")
            else:
                print("      [tip: /trainbrain first — then /export also writes a "
                      "llama.cpp-runnable gpt2 GGUF]")
        except Exception as ex:                      # noqa: BLE001 — demo surface
            print(f"      [export failed: {ex}]")
        return False
    if cmd == "/brain":
        nlm = mind.neural_language_model
        print(f"mind> My deep language model is a {nlm.cfg.layers}-layer transformer "
              f"({nlm.cfg.dim}d, {nlm.cfg.heads} heads, context {nlm.cfg.context_len}, "
              f"{nlm.n_params} parameters).")
        if nlm.is_trained:
            print(f"      Trained over {nlm.epochs_trained} epoch(s) so far, loss {nlm.last_loss:.3f} "
                  f"nats/token, {nlm.vocabulary_size} word(s) in my neural vocabulary, "
                  f"{nlm.tokens_observed} token(s) observed.")
        else:
            print(f"      Not trained yet — {nlm.buffered_sentences} sentence(s) waiting. "
                  "Type /trainbrain to deep-train me.")
        print(f"      My statistical n-gram model knows {mind.language_model.vocabulary} word(s) "
              f"from {mind.language_model.total_tokens} token(s).")
        return False
    if cmd == "/vocab":
        lex = mind.learner.lexicon if mind.learner else None
        if lex is None:
            print("      [learning is disabled — no vocabulary to show]")
            return False
        print(f"mind> I have picked up {lex.size} word(s) from talking with you "
              f"and own {mind.phrases.count} saying(s) in my own voice.")
        recent = lex.recent_new_words(8)
        if recent:
            print(f"      Newest: {', '.join(recent)}")
        return False
    if cmd == "/sayings":
        all_sayings = mind.phrases.all()
        print(f"mind> I know {len(all_sayings)} saying(s) I can speak in my own voice.")
        for s in all_sayings[:5]:
            print(f"      - {s}")
        if len(all_sayings) > 5:
            print(f"      ... and {len(all_sayings) - 5} more.")
        return False
    if cmd == "/gender" and rest:
        try:
            mind.set_gender(rest)
        except ValueError as exc:
            print(f"      [{exc}]")
            return False
        print(f"      [identity now: '{mind.gender_identity or '(unset)'}' — pronouns {mind.pronouns}.]")
        print(f"mind> {mind.describe_self()}")
        return False
    if cmd == "/gender":
        print("      [usage: /gender <identity | she/her | he/him>]")
        return False
    if cmd == "/pronouns" and rest:
        from .core.gender import parse_pronouns
        parsed = parse_pronouns(rest)
        if parsed is None:
            if rest.lower() == "they/them":
                print("      [they/them is reserved for describing people whose gender "
                      "the mind does not know — it is not a setting for its own "
                      "pronouns. Choose she/her or he/him.]")
            else:
                print("      [Unknown pronoun set. Known presets: she/her, he/him.]")
            return False
        try:
            mind.set_pronouns(parsed)
        except ValueError as exc:
            print(f"      [{exc}]")
            return False
        print(f"      [pronouns now {mind.pronouns}.]")
        print(f"mind> {mind.describe_self()}")
        return False
    if cmd == "/pronouns":
        print("      [usage: /pronouns <she/her | he/him>]")
        return False
    if cmd == "/who":
        print(f"mind> {mind.describe_self()}")
        return False

    # ---- feelings + internals + thinking controls (developer switches) ----
    if cmd == "/feeling":
        print(f"[mind.feeling.current_status] {mind.feeling.current_status}")
        return False
    if cmd == "/feelings":
        show = rest.lower() not in ("off", "false", "0")
        mind.expression.feelings(show)
        print("[feelings will be ANNOUNCED in chat again: '(my current read: ...)']"
              if show else
              "[feelings now color its words naturally — the raw state stays on /feeling]")
        return False
    if cmd == "/followups":
        ask = rest.lower() not in ("off", "false", "0")
        mind.expression.follow_ups(ask)
        print("[it may ask its own follow-up questions again]" if ask
              else "[it will only answer now — no Pick(...) questions]")
        return False
    if cmd == "/thinking":
        if rest.lower() in ("on", "true", "1"):
            mind.thinking_options.enable(True)
        elif rest.lower() in ("off", "false", "0"):
            mind.thinking_options.enable(False)
        o = mind.thinking_options
        print(f"[thinking enabled: {o.enabled} — level {o.level.value}, "
              f"ceiling {o.ceiling.value}, tone '{o.tone}']")
        return False
    if cmd == "/showthinking":
        show = rest.lower() not in ("off", "false", "0")
        mind.expression.thinking(show)
        print("[the thought trace will be printed in chat]" if show
              else "[thoughts hidden again — still on mind.last_thoughts / response.thoughts]")
        return False
    if cmd == "/level" and rest:
        from .core.options import ThinkingLevel
        if rest.lower().replace("-", "").replace("_", "") in (lv.value for lv in ThinkingLevel):
            requested = ThinkingLevel.parse(rest)
            mind.thinking_options.at(rest)
            o = mind.thinking_options
            clamp = "" if o.level == requested else f" (clamped by ceiling {o.ceiling.value})"
            print(f"[thinking level: {o.level.value}{clamp} → {o.thought_rounds} "
                  "generated thought(s) per cycle]")
        else:
            print("[unknown level. low | medium | high | veryhigh | max]")
        return False
    if cmd == "/ceiling" and rest:
        from .core.options import ThinkingCeiling
        c = ThinkingCeiling.parse(rest)
        key = rest.lower().replace("-", "").replace("_", "").replace(" ", "")
        if key.startswith("only"):
            key = key[4:]
        if key.isdigit() and 1 <= int(key) <= 5:
            mind.thinking_options.up_to(c)
            o = mind.thinking_options
            ladder = {"only1": "Low", "only2": "Low, Medium", "only3": "Low, Medium, High",
                      "only4": "Low, Medium, High, Very High",
                      "only5": "Low, Medium, High, Very High, Max"}[c.value]
            print(f"[thinking ceiling: {c.value} — allowed: {ladder}; "
                  f"current level {o.level.value} → {o.thought_rounds} thought(s) per cycle]")
        else:
            print("[unknown ceiling. only1..only5 (or 1..5): 1=Low only, 2=+Medium, 3=+High, 4=+VeryHigh, 5=+Max]")
        return False
    if cmd == "/tone" and rest:
        mind.thinking_options.with_tone(rest)
        print(f"[thinking tone: '{mind.thinking_options.tone}']")
        return False
    if cmd == "/meta":
        show = rest.lower() not in ("off", "false", "0")
        mind.expression.meta(show)
        print("[response meta (action/emotion/confidence) will be shown]" if show
              else "[response meta hidden]")
        return False
    if cmd in ("/exit", "/quit", "/bye"):
        return True
    print(f"[Unknown slash command '{cmd}'. Type /help for the list — plain sentences still just talk to it.]")
    return False


def _free_chat(mind) -> None:
    print("\n  Free chat with the mind. 'quit' to exit.")
    print("  Slash commands: /help — control, not conversation; plain talk goes to the mind.\n")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        if line.startswith("/"):
            if _handle_slash(line, mind):
                break
            continue
        low = line.lower()
        if low.startswith("train "):
            report = mind.train_file(line[6:].strip())
            print(f"mind> trained: {report}")
            print(f"      [vocabulary: {mind.phrases.count} saying(s) it can speak now]")
            continue
        if low.startswith("teach "):
            report = mind.train(line[6:].strip())
            print(f"mind> {report}")
            continue
        if low == "sayings":
            all_sayings = mind.phrases.all()
            print(f"mind> I know {len(all_sayings)} saying(s) I can speak in my own voice.")
            for s in all_sayings[:5]:
                print(f"      - {s}")
            if len(all_sayings) > 5:
                print(f"      ... and {len(all_sayings) - 5} more.")
            continue
        if low == "vocab":
            lex = mind.learner.lexicon if mind.learner else None
            if lex is None:
                print("      [learning is disabled — no vocabulary to show]")
                continue
            print(f"mind> I have picked up {lex.size} word(s) from talking with you "
                  f"and own {mind.phrases.count} saying(s) in my own voice.")
            recent = lex.recent_new_words(8)
            if recent:
                print(f"      Newest: {', '.join(recent)}")
            continue
        if low.startswith("gender "):
            word = line[7:].strip()
            if not word:
                print("      [usage: gender <identity | she/her | he/him>]")
                continue
            try:
                mind.set_gender(word)
            except ValueError as exc:
                print(f"      [{exc}]")
                continue
            print(f"      [identity now: '{mind.gender_identity or '(unset)'}' — "
                  f"pronouns {mind.pronouns}.]")
            print(f"mind> {mind.describe_self()}")
            continue
        if low.startswith("pronouns "):
            from .core.gender import parse_pronouns
            raw = line[9:].strip()
            parsed = parse_pronouns(raw)
            if parsed is None:
                if raw.lower() == "they/them":
                    print("      [they/them is reserved for describing people whose gender "
                          "the mind does not know — it is not a setting for its own "
                          "pronouns. Choose she/her or he/him.]")
                else:
                    print("      [Unknown pronoun set. Known presets: she/her, he/him.]")
                continue
            try:
                mind.set_pronouns(parsed)
            except ValueError as exc:
                print(f"      [{exc}]")
                continue
            print(f"      [pronouns now {mind.pronouns}.]")
            print(f"mind> {mind.describe_self()}")
            continue
        if low == "who are you":
            print(f"mind> {mind.describe_self()}")
            continue
        lex = mind.learner.lexicon if mind.learner else None
        before = lex.size if lex else 0
        r = mind.process(line)
        print(f"mind> {r.text}")
        # Internal detail stays INVISIBLE by default — the developer decides
        # what surfaces (--show-internals, or /showthinking + /meta):
        if mind.expression.show_thinking and mind.last_thoughts:
            print("      ~ its thoughts before speaking:")
            for t in mind.last_thoughts:
                print(f"        ~ {t}")
        if mind.expression.show_response_meta:
            print(f"      [action={r.meta.get('action')} "
                  f"emotion={r.meta.get('dominant_emotion')} "
                  f"conf={r.confidence:.2f}]")
        if lex and lex.size > before:
            print(f"      [picked up the word(s): {', '.join(lex.last_picks)} — "
                  f"vocabulary now {lex.size} word(s)]")


def main() -> None:
    argv = sys.argv[1:]
    args = set(argv)
    mind = make_full_mind().build()

    # Developer switches (presentation + thinking) — the mind never flips
    # its own:
    #   --thinking <low|medium|high|veryhigh|max>   thinking level
    #   --thinking-ceiling <only1..only5|1..5>      hard cap on thinking
    #   --thinking-tone <balanced|focused|playful|poetic|plain>
    #   --no-thinking        this model cannot think before speaking
    #   --show-feelings      announce feelings mechanically ("my current read: ...")
    #   --show-internals     show the thought trace + response meta in chat
    #   --no-followups       it answers but never asks its own questions
    for i, arg in enumerate(argv):
        if arg == "--thinking" and i + 1 < len(argv):
            mind.thinking_options.at(argv[i + 1])
        elif arg == "--thinking-ceiling" and i + 1 < len(argv):
            mind.thinking_options.up_to(argv[i + 1])
        elif arg == "--thinking-tone" and i + 1 < len(argv):
            mind.thinking_options.with_tone(argv[i + 1])
        elif arg == "--no-thinking":
            mind.thinking_options.enable(False)
        elif arg == "--show-feelings":
            mind.expression.feelings(True)
        elif arg == "--show-internals":
            mind.expression.thinking(True).meta(True)
        elif arg == "--no-followups":
            mind.expression.follow_ups(False)
    o, x = mind.thinking_options, mind.expression
    print(f"[thinking: {o.enabled}, level {o.level.value}, ceiling {o.ceiling.value}, "
          f"tone '{o.tone}' | feelings shown: {x.show_feeling_reads} | follow-ups: "
          f"{x.ask_follow_up_questions} | internals: {x.show_thinking}]")

    if "--all" in args:
        run_all()
        return
    # optional startup training: python -m cognify --train corpus/mind-starter.txt
    #   (+ corpus/words-and-meanings.txt for a 1,127-word recallable dictionary)
    for i, arg in enumerate(argv):
        if arg in ("--train", "-t") and i + 1 < len(argv):
            report = mind.train_file(argv[i + 1])
            print(f"[Trained from {argv[i + 1]}: {report}]")
            print(f"[Vocabulary: {mind.phrases.count} saying(s) it can speak now.]")
    print(f"\nCognifyPy - mind '{mind.name}' is awake. Choose a section:")
    while True:
        for key, (label, _) in _MENU.items():
            print(f"  {key}. {label}")
        try:
            choice = input("\nselect> ").strip()
        except EOFError:
            break
        if choice.lower() in ("q", "quit", "exit"):
            break
        if choice == "8":
            _free_chat(mind)
        elif choice in _MENU:
            _MENU[choice][1](mind)
        else:
            print("  (choose a number, or 'q' to quit)")
    print("\nGoodbye - the mind sleeps.")


if __name__ == "__main__":
    main()
