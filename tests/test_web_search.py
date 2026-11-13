"""Tests for the manual-only web search capability."""
import pytest

from cognify.capabilities.web_search import (
    GoogleNewsBackend,
    MultiEngineBackend,
    WebSearchError,
    WebSearchResult,
    WikipediaBackend,
    rank_by_focus,
    search_web,
)
from cognify.core.builder import MindBuilder


class FakeBackend:
    """Deterministic offline backend."""

    def __init__(self, fail=False, results=None):
        self.calls = 0
        self.last_query = ""
        self.fail = fail
        self.results = results or [
            WebSearchResult(
                title="Quokka",
                url="https://en.wikipedia.org/wiki/Quokka",
                snippet=("The quokka is a small macropod about the size of a "
                         "domestic cat. It is the only terrestrial mammal on "
                         "Rottnest Island."),
            ),
        ]

    def search(self, query, max_results):
        self.calls += 1
        self.last_query = query
        if self.fail:
            raise WebSearchError("web search could not complete (offline)")
        return self.results[:max_results]


def make_mind(backend):
    mind = (MindBuilder("Curious")
            .with_preset("curious_child")
            .enable_tool_use()
            .build())
    mind.web_search_backend = backend
    return mind


def test_search_web_returns_results():
    backend = FakeBackend()
    mind = make_mind(backend)
    output = mind.search_web("what is a quokka")
    assert backend.calls == 1
    assert "quokka" in output.lower()


def test_search_web_learns_memories_facts_and_language():
    backend = FakeBackend()
    mind = make_mind(backend)

    mind.search_web("what is a quokka")

    assert any("quokka" in m.content.lower()
               for m in mind.memory.all())
    assert mind.kb.size() > 0  # snippet facts distilled
    assert mind.language_model.is_trained


def test_search_web_marks_query_words_known():
    backend = FakeBackend()
    mind = make_mind(backend)
    lex = mind.learner.lexicon
    lex.observe("quokka", 1)
    assert lex.fresh_word() is not None  # would normally be asked about

    mind.search_web("what is a quokka")

    assert lex.fresh_word() is None  # looked up instead of asked


def test_search_web_never_auto_fires_from_conversation():
    backend = FakeBackend()
    mind = make_mind(backend)

    mind.process("can you search the web for quokkas?")
    mind.process("web search would be useful now")
    mind.process("use tool web-search query=quokka")  # tool registry has no web tool

    assert backend.calls == 0


def test_search_web_reports_failure_cleanly():
    backend = FakeBackend(fail=True)
    mind = make_mind(backend)
    output = mind.search_web("anything")
    assert output.startswith("I could not search the web")
    assert "internet connection" in output


def test_search_web_empty_query():
    mind = make_mind(FakeBackend())
    assert "No search query" in mind.search_web("   ")


def test_google_news_parse_extracts_results():
    page = """
    <rss><channel>
    <item><title>Quokka boom in the news</title>
      <link>https://news.google.com/rss/articles/example</link>
      <source url="https://example.org">Example News</source>
      <description><![CDATA[Short and clean.]]></description></item>
    </channel></rss>
    """
    results = GoogleNewsBackend.parse(page, 4)
    assert len(results) == 1
    assert "Quokka boom in the news" in results[0].title
    assert "Example News" in results[0].snippet


def test_wikipedia_parse_extracts_results():
    page = ('{"query": {"search": [{"title": "Quokka", "snippet": '
            '"The quokka is a <span class=searchmatch>macropod</span>."}]}}')
    results = WikipediaBackend.parse(page, 4)
    assert len(results) == 1
    assert results[0].url.endswith("/wiki/Quokka")
    assert "macropod" in results[0].snippet


def test_relevance_gate_rejects_off_topic_results():
    backend = MultiEngineBackend(engines=[])
    junk = [WebSearchResult(title="Pizza near you",
                            url="https://pizza.example/",
                            snippet="Best pizza in town.")]
    assert not backend.is_relevant(["quokka"], junk)
    assert backend.is_relevant([], junk)  # no content words → pass


def test_multi_engine_backend_topic_retry(monkeypatch):
    class JunkEngine:
        name = "Junk"
        calls = []

        def fetch(self, query, max_results):
            JunkEngine.calls.append(query)
            return [WebSearchResult(title="junk page",
                                    url="https://junk.example/x",
                                    snippet="completely unrelated words")]

    class GoodEngine:
        name = "Good"

        def fetch(self, query, max_results):
            return [WebSearchResult(title=f"quokka facts ({query})",
                                    url="https://wiki.example/quokka",
                                    snippet="the quokka is a macropod")]

    backend = MultiEngineBackend(engines=[JunkEngine(), GoodEngine()])
    results = backend.search("wikipedia quokka", 4)
    # The first pass failed the relevance gate ("quokka" missing from the
    # junk), so the topic retry ran with the substantial words only.
    assert len(results) == 1
    assert "quokka" in results[0].title
    assert any("quokka" in q for q in JunkEngine.calls)


def test_rank_by_focus_reorders():
    hits = [
        WebSearchResult(title="Quokka habitat", url="https://a.example/q",
                        snippet="Quokkas live on small islands."),
        WebSearchResult(title="Quokka diet", url="https://b.example/qd",
                        snippet="Quokkas eat grass and leaves. The quokka diet is mostly plants."),
    ]
    ranked = rank_by_focus(hits, "what do quokkas eat")
    assert ranked[0].title == "Quokka diet"
    assert rank_by_focus(hits, None)[0].title == "Quokka habitat"


def test_search_web_focus_steers_learning():
    backend = FakeBackend(results=[
        WebSearchResult(title="Quokka habitat", url="https://a.example/q",
                        snippet="Quokkas live on small islands off the coast."),
        WebSearchResult(title="Quokka diet", url="https://b.example/qd",
                        snippet="Quokkas eat grass and leaves. The quokka diet is mostly plants."),
    ])
    mind = make_mind(backend)
    output = mind.search_web("quokkas", focus="what do quokkas eat")
    assert backend.last_query == "quokkas"      # quoted part stays the query
    assert "focusing on" in output
    memories = [m for m in mind.memory.all() if "web" in m.tags]
    assert memories and "Quokka diet" in memories[0].content


def test_web_fetch_parse_and_links():
    from cognify.capabilities.web_fetch import extract_links, parse_page
    html = ("<html><head><title>Apple</title></head><body>"
            "<p>An apple is a sweet fruit that grows on trees.</p>"
            "<script>ignore()</script>"
            '<a href="/wiki/Blossom">Blossom</a>'
            '<a href="https://other.example/x">External</a>'
            "</body></html>")
    title, text = parse_page(html)
    assert title == "Apple"
    assert "sweet fruit" in text and "ignore()" not in text
    links = extract_links(html, "https://en.wikipedia.org/wiki/Apple")
    assert len(links) == 1 and links[0][0].endswith("/wiki/Blossom")


def test_backend_protocol_search_function():
    backend = FakeBackend()
    results = search_web("hello", backend=backend, max_results=1)
    assert len(results) == 1
    assert results[0].title == "Quokka"
