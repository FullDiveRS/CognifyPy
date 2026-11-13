"""Web page reading + site exploration — her own WebFetch.

Mirrors the C# ``WebPageReader`` / ``WebExplorer``: fetch a page the way a
person reads (throw away scripts, styles and chrome, keep readable text plus
the title), then — for exploration — FOLLOW internal links (same host,
breadth first) so the mind is not locked to one page or one search engine.
When a focus suggestion is given ("look for any information on flowers"),
the links whose address/anchor text best match the focus words are followed
first — the exploration steers toward what was asked.

Failures are classified with the same honest categories as search
(offline / blocked / nothing readable).
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .web_search import _fetch, WebSearchError, WebSearchFailure

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA,
            "Accept-Language": "en-US,en;q=0.9"}

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>",
                       re.IGNORECASE | re.DOTALL)
_ANCHOR_RE = re.compile(
    r"<a[^>]*href=\"([^\"#]+)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")

_SKIP_SUFFIXES = (".png", ".jpg", ".gif", ".css", ".js", ".ico")


@dataclass
class WebPage:
    """One fetched web page: its url, title and readable text."""
    url: str
    title: str
    text: str

    def excerpt(self, max_chars: int = 260) -> str:
        if len(self.text) <= max_chars:
            return self.text
        return self.text[:max_chars].rstrip() + "..."


def _clean(fragment: str) -> str:
    return _WS_RE.sub(" ", fragment).strip()


def parse_page(html: str) -> Tuple[str, str]:
    """Extract (title, readable text) from raw HTML."""
    title_match = _TITLE_RE.search(html)
    title = _clean(_TAG_RE.sub(" ", title_match.group(1))) if title_match else ""
    title = _html.unescape(title)
    body = _SCRIPT_STYLE_RE.sub(" ", html)
    body = _COMMENT_RE.sub(" ", body)
    body = _TAG_RE.sub(" ", body)
    text = _html.unescape(body)
    return title, _clean(text)


# A descriptive bot User-Agent used as a FALLBACK. Some sites (Wikimedia
# among them) 403 the fake-Chrome UA from datacenter IPs but happily serve an
# honest, descriptive bot agent — their policy prefers identifiable clients.
_BOT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CognifyBot/1.0; "
                              "+https://github.com/cognify-ai)",
                "Accept-Language": "en-US,en;q=0.9"}


def _fetch_html(url: str, timeout: float = 12.0) -> str:
    """Fetch the raw HTML of one page with honest failure classification.
    When the site REFUSES the browser-like client (403/429), the read is
    retried once with the descriptive bot agent — same page, honest UA."""
    try:
        return _fetch(url, timeout, engine="web-fetch")
    except WebSearchError as exc:
        if exc.reason is not WebSearchFailure.BLOCKED:
            raise
        request = urllib.request.Request(url, headers=_BOT_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "ignore")
        except Exception as retry_exc:
            raise WebSearchError(
                f"web-fetch could not be reached ({retry_exc})",
                WebSearchFailure.OFFLINE) from retry_exc


def read_page(url: str, timeout: float = 12.0) -> Tuple[str, str]:
    """Fetch one page and extract (title, readable text). Raises
    :class:`WebSearchError` with an honest reason on failure."""
    html = _fetch_html(url, timeout)
    title, text = parse_page(html)
    if len(text) < 80:
        raise WebSearchError(
            "the page had no readable text (scripts-only or a bot-check)",
            WebSearchFailure.BLOCKED)
    return title, text


def extract_links(html: str, page_url: str,
                  host: Optional[str] = None) -> List[Tuple[str, str]]:
    """Internal links of a page as (absoluteUrl, anchorText), same-host only,
    deduped, anchors stripped of tags."""
    try:
        base = urllib.parse.urlsplit(page_url)
    except ValueError:
        return []
    host = host or base.netloc
    seen: set = set()
    links: List[Tuple[str, str]] = []
    for match in _ANCHOR_RE.finditer(html):
        href = _html.unescape(match.group(1)).strip()
        if not href or href.lower().startswith(("javascript:", "mailto:")):
            continue
        resolved = urllib.parse.urljoin(page_url, href)
        parts = urllib.parse.urlsplit(resolved)
        if parts.scheme not in ("http", "https"):
            continue
        if parts.netloc.lower() != host.lower():
            continue
        path = parts.path.lower()
        if path.endswith(_SKIP_SUFFIXES):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        anchor = _clean(_TAG_RE.sub(" ", match.group(2)))
        links.append((resolved, anchor))
    return links


def _content_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']{3,}", (text or "").lower())


class WebExplorer:
    """The site explorer behind ``/explore`` and ``mind.explore_url``:
    reads a starting page, then FOLLOWS internal links (same host, breadth
    first). Focus words (from the user's suggestion) rank which links get
    followed. One unreadable page never kills the exploration."""

    def __init__(self, timeout: float = 12.0, max_pages: int = 3) -> None:
        self.timeout = timeout
        self.max_pages = max(1, min(6, max_pages))

    def explore(self, start_url: str,
                focus: Optional[str] = None) -> List[WebPage]:
        parts = urllib.parse.urlsplit(start_url or "")
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise WebSearchError(f"not a web url: {start_url}",
                                 WebSearchFailure.NO_RESULTS)

        focus_words = _content_words(focus or "")
        pages: List[WebPage] = []
        visited: set = set()
        queue: List[str] = [start_url]

        while queue and len(pages) < self.max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                html = _fetch_html(url, self.timeout)
                title, text = parse_page(html)
            except WebSearchError:
                continue      # one unreadable page never kills the exploration
            if len(text) < 80:
                continue
            pages.append(WebPage(url=url, title=title, text=text))
            if len(pages) >= self.max_pages:
                break

            links = [(u, a) for u, a in extract_links(html, url)
                     if u not in visited]
            if focus_words:
                def link_score(link: Tuple[str, str]) -> int:
                    hay = f"{link[0]} {link[1]}".lower()
                    return sum(1 for w in focus_words if w in hay)
                links.sort(key=link_score, reverse=True)
            for link_url, _ in links[:4]:
                queue.append(link_url)

        return pages
