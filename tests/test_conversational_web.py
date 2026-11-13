"""Tests for the conversational + web batch (mirror of the C#
ConversationalWebTests): the focus suggestion, explore_url, the knowledge-gap
AUTO lookup (with_auto_web_fetch), THINKING AUTOMATIC and the Conversational
options (rate → chattier replies, understand → topic-shift notes)."""
import pytest

from cognify.capabilities.web_search import (
    WebSearchError,
    WebSearchResult,
    rank_by_focus,
)
from cognify.core.builder import MindBuilder


class FakeBackend:
    def __init__(self, results=None, fail=False):
        self.calls = 0
        self.last_query = ""
        self.fail = fail
        self.results = results or []

    def search(self, query, max_results):
        self.calls += 1
        self.last_query = query
        if self.fail:
            raise WebSearchError("web search could not complete (offline)")
        return self.results[:max_results]


TWO_HITS = [
    WebSearchResult("Quokka habitat", "https://a.example/quokka",
                    "Quokkas live on small islands off the coast."),
    WebSearchResult("Quokka diet", "https://b.example/quokka-diet",
                    "Quokkas eat grass and leaves. The quokka diet is mostly plants."),
]


def _build(**kw):
    builder = (MindBuilder("Curious")
               .with_preset("curious_child")
               .with_reasoning()
               .with_memory())
    if kw.pop("auto", False):
        builder = builder.with_auto_web_fetch(**kw)
    return builder.build()


# ------------------------------------------------------------------ focus
def test_search_web_focus_reorders_reported_results():
    backend = FakeBackend(results=TWO_HITS)
    mind = _build()
    mind.web_search_backend = backend
    output = mind.search_web("quokkas", focus="what do quokkas eat")
    assert backend.last_query == "quokkas"
    assert output.index("Quokka diet") < output.index("Quokka habitat")


def test_search_web_focus_learns_focused_hit_first():
    backend = FakeBackend(results=TWO_HITS)
    mind = _build()
    mind.web_search_backend = backend
    mind.search_web("quokkas", focus="what do quokkas eat")
    memories = [m for m in mind.memory.all() if "web" in m.tags]
    assert memories and "Quokka diet" in memories[0].content


# ------------------------------------------------------------------ explore
def test_explore_url_invalid_fails_gracefully():
    mind = _build()
    output = mind.explore_url("not-a-url", focus="flowers")
    assert "not a web url" in output.lower()


def test_explore_url_unreachable_fails_gracefully():
    mind = _build()
    output = mind.explore_url("https://cognify-tests.invalid/page", focus="flowers")
    assert "could not read anything" in output.lower() or \
        "could not explore" in output.lower()


# ------------------------------------------------------------------ automatic
def _chain_texts(mind, text):
    response = mind.process(text)
    return [t[1] if isinstance(t, tuple) else t.content
            for t in response.thoughts]


def test_thinking_automatic_short_smalltalk_thinks_low():
    mind = _build()
    mind.thinking_options.automatic_on()
    mind.process("hi")            # first "hi" is a novelty surprise — Medium
    texts = _chain_texts(mind, "hi")   # familiar line now → quick Low think
    assert any("chose to think at low" in t for t in texts)


def test_thinking_automatic_long_why_question_thinks_hard():
    mind = _build()
    mind.thinking_options.automatic_on()
    texts = _chain_texts(
        mind,
        "Why does the sky look blue in the day and red at sunset, and how "
        "does the light scatter through the atmosphere to make those colors happen?")
    assert any("chose to think at veryhigh" in t or "chose to think at max" in t
               for t in texts)


def test_thinking_automatic_respects_ceiling():
    mind = _build()
    mind.thinking_options.automatic_on()
    mind.thinking_options.up_to("only1")
    texts = _chain_texts(
        mind,
        "Why does the sky look blue in the day and red at sunset, and how "
        "does the light scatter through the atmosphere to make those colors happen?")
    assert any("chose to think at low" in t for t in texts)


def test_thinking_automatic_off_keeps_fixed_level():
    mind = _build()
    mind.thinking_options.automatic_on(False)
    texts = _chain_texts(
        mind,
        "Why does the sky look blue in the day and red at sunset, and how "
        "does the light scatter through the atmosphere to make those colors happen?")
    assert not any("Automatic:" in t for t in texts)


# ------------------------------------------------------------------ conversational
def test_conversational_understand_remembers_topic_shifts():
    mind = _build()
    mind.conversational.understand = True
    mind.process("tell me about quokkas")
    mind.process("let us talk about music now")
    notes = [m for m in mind.memory.all() if "conversation" in m.tags]
    assert notes and "quokkas" in notes[0].content and "music" in notes[0].content


def test_conversational_understand_off_no_topic_notes():
    mind = _build()
    mind.conversational.understand = False
    mind.process("tell me about quokkas")
    mind.process("let us talk about music now")
    assert not [m for m in mind.memory.all() if "conversation" in m.tags]


def test_conversational_rate_makes_replies_longer():
    inputs = [
        "tell me about quokkas",
        "let us talk about music",
        "I love painting",
        "rain is nice today",
        "books make me happy",
        "tell me about oceans",
    ]

    def total_words(rate):
        mind = _build()
        mind.conversational.rate = rate
        return sum(len(mind.process(line).text.split()) for line in inputs)

    assert total_words(2.0) > total_words(1.0)


# ------------------------------------------------------------------ auto fetch
def test_auto_web_fetch_unknown_topic_searches_learns_answers():
    backend = FakeBackend(results=TWO_HITS)
    mind = _build(auto=True, backend=backend)
    response = mind.process("what is a quokka?")
    assert backend.calls == 1
    assert "searched the web" in response.text
    assert "Quokka" in response.text
    assert any("web" in m.tags for m in mind.memory.all())


def test_auto_web_fetch_same_topic_looks_up_only_once():
    backend = FakeBackend(results=TWO_HITS)
    mind = _build(auto=True, backend=backend)
    mind.process("what is a quokka?")
    assert backend.calls == 1
    mind.process("what is a quokka?")
    assert backend.calls == 1


def test_auto_web_fetch_known_topic_never_searches():
    backend = FakeBackend(results=TWO_HITS)
    mind = _build(auto=True, backend=backend)
    mind.teach("quokkas are marsupials")
    response = mind.process("what is a quokka?")
    assert backend.calls == 0
    assert "searched the web" not in response.text


def test_auto_web_fetch_personal_and_math_never_searches():
    backend = FakeBackend(results=TWO_HITS)
    mind = _build(auto=True, backend=backend)
    mind.process("what do you like to talk about?")
    mind.process("what is 2 + 2?")
    assert backend.calls == 0


def test_auto_web_fetch_failed_lookup_stays_silent():
    backend = FakeBackend(results=TWO_HITS, fail=True)
    mind = _build(auto=True, backend=backend)
    response = mind.process("what is a quokka?")
    assert backend.calls == 1
    assert "searched the web" not in response.text


def test_auto_web_fetch_off_never_searches():
    backend = FakeBackend(results=TWO_HITS)
    mind = _build()
    mind.web_search_backend = backend   # manual-only, no auto
    response = mind.process("what is a quokka?")
    assert backend.calls == 0
    assert "searched the web" not in response.text


# ------------------------------------------------------------------ persistence
def test_snapshot_roundtrip_keeps_new_options(tmp_path):
    mind = _build(auto=True)
    mind.thinking_options.automatic_on()
    mind.conversational.rate_amount(2.0)
    data = mind.to_dict()
    restored = Mind.from_dict_static(data) if hasattr(Mind, "from_dict_static") \
        else _restore(data)
    assert restored.auto_fetch_web
    assert restored.thinking_options.automatic
    assert restored.conversational.rate == 2.0


def _restore(data):
    from cognify.core.mind import Mind
    return Mind.from_dict(data)


from cognify.core.mind import Mind  # noqa: E402  (used above)
