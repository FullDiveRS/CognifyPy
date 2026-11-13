"""Web search — a MANUAL-ONLY capability.

The mind never searches on its own: there is no cycle hook here, no
capability that fires on text. The ONLY paths are:

  * ``mind.search_web(query, focus=None)`` (library API), and
  * the demo's ``/searchweb "<query>" [suggestion]`` slash command.

Both are explicit user decisions (or, with ``with_auto_web_fetch()`` an
explicit builder decision — the auto path runs through the same backend).

The default backend is a multi-engine chain (Google News RSS → Wikipedia
API → Bing RSS; DuckDuckGo was removed by design). Two guards keep the
results honest:

1. THE RELEVANCE GATE. An engine's output is only accepted when the results
   actually MATCH the query: every content word of the query (3+ letters)
   must appear in at least one result's title/snippet/host. This is the fix
   for "wikipedia flowers" returning nothing but Wikipedia-site pages (the
   word "flowers" silently vanished) — an engine that misses part of the
   question is skipped, the next one gets a try.
2. THE TOPIC RETRY. When every engine missed part of the query, the search
   retries once with the query's substantial topic words only, so a clunky
   query still finds its subject.

When everything fails the reported reason is honest — "offline" only when
the network truly failed.
"""
from __future__ import annotations

import base64
import html as _html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Protocol, Sequence


class WebSearchFailure(Enum):
    """Why a web search could not complete."""
    OFFLINE = "offline"          # the machine could not reach the network
    BLOCKED = "blocked"          # the network worked but the engine sent a bot-check
    NO_RESULTS = "no_results"    # everything answered; nothing readable matched


class WebSearchError(Exception):
    """Raised when a web search could not complete (offline, blocked, no parse)."""

    def __init__(self, message: str,
                 reason: WebSearchFailure = WebSearchFailure.OFFLINE) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class WebSearchResult:
    """One web search hit: title, url and snippet."""
    title: str
    url: str
    snippet: str


class WebSearchBackend(Protocol):
    """A pluggable search backend (multi-engine by default; plug in your own)."""

    def search(self, query: str, max_results: int) -> List[WebSearchResult]:
        ...


_TAG_RE = re.compile(r"<[^>]+>")
# A browser-like User-Agent: search engines bot-check unfamiliar clients,
# which is exactly what broke the old "CognifyMind/1.0" client.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}


def _strip_tags(fragment: str) -> str:
    text = _TAG_RE.sub(" ", fragment)
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _unwrap_url(url: str) -> str:
    # DuckDuckGo links are redirects: /l/?uddg=<encoded real url> — unwrap.
    idx = url.find("uddg=")
    if idx >= 0:
        encoded = url[idx + 5:]
        end = encoded.find("&")
        if end > 0:
            encoded = encoded[:end]
        return urllib.parse.unquote(encoded)
    return url


def _unwrap_bing_url(url: str) -> str:
    """Bing wraps hits in /ck/a redirects whose u=a1... parameter is the
    base64url of the real destination; decode it, pass plain links through."""
    idx = url.find("u=a1")
    if idx < 0:
        return url
    token = url[idx + 4:]
    end = token.find("&")
    if end > 0:
        token = token[:end]
    try:
        padded = token + "=" * (-len(token) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", "replace")
    except Exception:
        return url


def _fetch(url: str, timeout: float, data: Optional[dict] = None,
           engine: str = "engine") -> str:
    """GET (or form POST when ``data``) with bot-check classification."""
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            page = response.read().decode("utf-8", "ignore")
            status = getattr(response, "status", 200)
    except Exception as exc:  # urllib raises a family of errors
        raise WebSearchError(f"{engine} could not be reached ({exc})",
                             WebSearchFailure.OFFLINE) from exc
    if status == 202 or status in (403, 429) or _is_anomaly_page(page):
        raise WebSearchError(f"{engine} answered with a bot-check page",
                             WebSearchFailure.BLOCKED)
    return page


def _is_anomaly_page(page: str) -> bool:
    """True when the page is a bot-check/challenge instead of results."""
    if len(page) < 200:
        return True
    return "anomaly" in page.lower() and "result" not in page.lower()


class _Engine(Protocol):
    name: str

    def fetch(self, query: str, max_results: int) -> List[WebSearchResult]:
        ...


class BingBackend:
    """Bing web search (third engine — general web when the others cannot
    answer). Uses Bing's RSS
    output (``format=rss``) — a small, stable, bot-resistant page with direct
    links and real snippets; the HTML scrape of b_algo blocks served
    randomized junk to lightweight clients. NOTE: Bing's RSS endpoint is
    known to serve locally-personalized junk to datacenter IPs — the
    backend's relevance gate catches that and moves on to the next engine."""

    name = "Bing"

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def fetch(self, query: str, max_results: int) -> List[WebSearchResult]:
        url = ("https://www.bing.com/search?q=" + urllib.parse.quote_plus(query) +
               f"&format=rss&count={max(10, max_results)}&mkt=en-US")
        page = _fetch(url, self.timeout, engine=self.name)
        return self.parse(page, max_results)

    @staticmethod
    def parse(page: str, max_results: int) -> List[WebSearchResult]:
        results: List[WebSearchResult] = []
        for item in re.findall(r"<item>(.*?)</item>", page,
                               re.IGNORECASE | re.DOTALL):
            if len(results) >= max_results:
                break

            def _text(tag: str) -> str:
                m = re.search(rf"<{tag}>(.*?)</{tag}>", item,
                              re.IGNORECASE | re.DOTALL)
                if not m:
                    return ""
                raw = m.group(1)
                cdata = re.match(r"<!\[CDATA\[(.*?)\]\]>$", raw, re.DOTALL)
                return _strip_tags(cdata.group(1) if cdata else raw)

            title, link = _text("title"), _text("link")
            if not title or not link:
                continue
            results.append(WebSearchResult(title=title, url=_unwrap_bing_url(link),
                                           snippet=_text("description")))
        return results


class GoogleNewsBackend:
    """Google web search (first engine in the default chain). Google's regular
    results page requires JavaScript, but Google NEWS offers a clean RSS feed
    (news.google.com/rss/search) — keyless, bot-resistant, and genuinely
    relevant. Items carry the real publisher in <source url="...">."""

    name = "Google"

    def __init__(self, timeout: float = 10.0, edition: str = "en-US") -> None:
        self.timeout = timeout
        self.edition = edition

    def fetch(self, query: str, max_results: int) -> List[WebSearchResult]:
        url = ("https://news.google.com/rss/search?q=" +
               urllib.parse.quote_plus(query) +
               f"&hl={urllib.parse.quote(self.edition)}&gl=US&ceid=US:en")
        page = _fetch(url, self.timeout, engine=self.name)
        return self.parse(page, max_results)

    @staticmethod
    def parse(page: str, max_results: int) -> List[WebSearchResult]:
        results: List[WebSearchResult] = []
        for item in re.findall(r"<item>(.*?)</item>", page,
                               re.IGNORECASE | re.DOTALL):
            if len(results) >= max_results:
                break

            def _text(tag: str) -> str:
                m = re.search(rf"<{tag}>(.*?)</{tag}>", item,
                              re.IGNORECASE | re.DOTALL)
                if not m:
                    return ""
                raw = m.group(1)
                cdata = re.match(r"<!\[CDATA\[(.*?)\]\]>$", raw, re.DOTALL)
                return _strip_tags(cdata.group(1) if cdata else raw)

            title, link = _text("title"), _text("link")
            if not title or not link:
                continue
            source = re.search(
                r'<source[^>]*url="([^"]+)"[^>]*>(.*?)</source>',
                item, re.IGNORECASE | re.DOTALL)
            publisher = _strip_tags(source.group(2)) if source else ""
            description = _text("description")
            # Snippet: the publisher name grounds the hit; the description is
            # usually a related-articles list, so it only fills in when short.
            if publisher:
                snippet = publisher + (" — " + description
                                       if 0 < len(description) < 160 else "")
            else:
                snippet = description
            results.append(WebSearchResult(title=title, url=link, snippet=snippet))
        return results


class WikipediaBackend:
    """Wikipedia search (second engine — the knowledge fallback). Wikipedia's
    public API is keyless, never bot-checked, and ideal for an AI that wants
    to UNDERSTAND a topic: results arrive as JSON with a clean text snippet
    and a stable article URL."""

    name = "Wikipedia"

    def __init__(self, timeout: float = 10.0, language: str = "en") -> None:
        self.timeout = timeout
        self.language = language

    def fetch(self, query: str, max_results: int) -> List[WebSearchResult]:
        url = (f"https://{self.language}.wikipedia.org/w/api.php"
               "?action=query&list=search&format=json&srlimit="
               f"{max(5, max_results)}&srsearch=" + urllib.parse.quote_plus(query))
        page = _fetch(url, self.timeout, engine=self.name)
        return self.parse(page, max_results, self.language)

    @staticmethod
    def parse(page: str, max_results: int, language: str = "en") -> List[WebSearchResult]:
        results: List[WebSearchResult] = []
        try:
            data = json.loads(page)
        except ValueError:
            return results
        hits = (data.get("query") or {}).get("search") or []
        for hit in hits:
            if len(results) >= max_results:
                break
            title = (hit.get("title") or "").strip()
            if not title:
                continue
            snippet = _strip_tags(hit.get("snippet") or "")
            slug = urllib.parse.quote(title.replace(" ", "_"))
            results.append(WebSearchResult(
                title=title,
                url=f"https://{language}.wikipedia.org/wiki/{slug}",
                snippet=snippet))
        return results


class MultiEngineBackend:
    """Tries each engine in order until one produces results that MATCH the
    query (the relevance gate), with one topic-retry when every engine missed
    part of a multi-word query. When every attempt fails, the reported reason
    is honest: OFFLINE only when the network truly failed, BLOCKED when
    engines sent bot-checks, NO_RESULTS when pages parsed cleanly but matched
    nothing."""

    # Query stop-words: dropped from the relevance gate and the topic retry.
    _STOP_WORDS = {
        "the", "and", "for", "with", "that", "this", "what", "who", "when",
        "where", "why", "how", "are", "was", "were", "you", "your", "about",
        "any", "can", "could", "does", "did", "from", "have", "has", "into",
        "look", "please", "search", "find", "tell", "give", "some", "info",
        "information", "she", "her", "his", "him", "its", "our", "out",
        "get", "got", "all", "not", "but", "just", "like", "want", "need",
    }

    def __init__(self, engines: Optional[Sequence[_Engine]] = None,
                 timeout: float = 10.0, attempts: int = 1) -> None:
        self.engines: Sequence[_Engine] = engines or (
            GoogleNewsBackend(timeout), WikipediaBackend(timeout),
            BingBackend(timeout))
        self.timeout = timeout
        self.attempts = attempts

    @classmethod
    def content_words(cls, query: str) -> List[str]:
        """The query's content words: 3+ letters, stop-words removed, deduped."""
        words = re.findall(r"[a-z0-9']{3,}", (query or "").lower())
        return list(dict.fromkeys(w for w in words
                                  if w not in cls._STOP_WORDS))

    @classmethod
    def is_relevant(cls, content_words: List[str],
                    results: List[WebSearchResult]) -> bool:
        """The relevance gate: every content word must show up in at least
        one result's title/snippet/host. Queries with no content words pass."""
        if not content_words:
            return True
        haystacks = [
            f"{r.title} {r.snippet} {_safe_host(r.url)}" for r in results]
        return all(any(w in h.lower() for h in haystacks)
                   for w in content_words)

    def search(self, query: str, max_results: int) -> List[WebSearchResult]:
        if not query or not query.strip():
            raise WebSearchError("Empty web search query.")
        saw_response = False   # at least one engine answered at HTTP level
        saw_blocked = False    # at least one bot-check
        detail = "no results"
        best_effort: Optional[List[WebSearchResult]] = None

        def collect(results, why):
            nonlocal best_effort, detail
            if results and best_effort is None:
                best_effort = results
            if why:
                detail = why

        words = self.content_words(query)
        results, why = self._run_once(query, max_results, words)
        collect(results, why)

        if best_effort is None and len(words) > 1:
            # Topic retry: keep the query's substantial words only.
            topic = " ".join(sorted(words, key=len, reverse=True)[:3])
            if topic.strip().lower() != query.strip().lower():
                results, why = self._run_once(topic, max_results,
                                              self.content_words(topic))
                collect(results, why)

        if best_effort is not None:
            return best_effort

        if saw_blocked:
            reason = WebSearchFailure.BLOCKED
        elif saw_response:
            reason = WebSearchFailure.NO_RESULTS
        else:
            reason = WebSearchFailure.OFFLINE
        raise WebSearchError(f"web search could not complete ({detail})", reason)

    def _run_once(self, query, max_results, words):
        saw_response = False
        saw_blocked = False
        detail = "no results"
        best: Optional[List[WebSearchResult]] = None
        for _ in range(max(1, self.attempts)):
            for engine in self.engines:
                try:
                    results = engine.fetch(query, max_results)
                    if not results:
                        saw_response = True
                        detail = (f"{engine.name} returned a page with no "
                                  "readable results")
                        continue
                    saw_response = True
                    if best is None:
                        best = results   # keep as last-resort output
                    if self.is_relevant(words, results):
                        return results, ""
                    detail = f"{engine.name} results did not match the full query"
                except WebSearchError as exc:
                    if exc.reason == WebSearchFailure.BLOCKED:
                        saw_response = True
                        saw_blocked = True
                    elif exc.reason == WebSearchFailure.NO_RESULTS:
                        saw_response = True
                    detail = str(exc)
        if saw_blocked and not detail:
            detail = "blocked"
        return None, detail


def _safe_host(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc or ""
    except ValueError:
        return ""


def rank_by_focus(results: List[WebSearchResult],
                  focus: Optional[str]) -> List[WebSearchResult]:
    """Re-rank results by the focus suggestion: hits whose title/snippet
    mention more of the focus's content words come first. Stable — equal
    scores keep their engine order. No focus (or no content words) leaves
    the order untouched."""
    words = MultiEngineBackend.content_words(focus or "")
    if not words or len(results) <= 1:
        return results

    def score(r: WebSearchResult) -> int:
        hay = f"{r.title} {r.snippet}".lower()
        return sum(1 for w in words if w in hay)

    return sorted(results, key=lambda r: -score(r))


def search_web(query: str, backend: WebSearchBackend | None = None,
               max_results: int = 4) -> List[WebSearchResult]:
    """Search the web and return up to max_results hits.
    Raises :class:`WebSearchError` when the search cannot complete."""
    backend = backend or MultiEngineBackend()
    return backend.search(query, max_results)
