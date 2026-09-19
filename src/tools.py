"""
The agent's tool set. Three tools, each returning a ToolResult so failures
are data, not exceptions -- the graph can see "this tool failed" or
"this tool returned nothing" and route accordingly instead of crashing.

  - web_search(query)      : DuckDuckGo search -> list of {title, url, snippet}
  - fetch_page(url)        : download + extract main text from a page
  - summarise(text, q)     : condense fetched text into a note focused on the question

Every tool also has an offline/fixture path (used when offline=True) so the
whole graph can be exercised without network access or an API key -- see
fixtures/ and README.md "Offline demo mode".
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    empty: bool = False  # ok=True but nothing useful came back

    @staticmethod
    def success(data: Any, empty: bool = False) -> "ToolResult":
        return ToolResult(ok=True, data=data, empty=empty)

    @staticmethod
    def failure(error: str) -> "ToolResult":
        return ToolResult(ok=False, error=error)


# --------------------------------------------------------------------------
# web_search
# --------------------------------------------------------------------------

def _url_slug(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def web_search(query: str, max_results: int = 4, offline: bool = False) -> ToolResult:
    if not query or not query.strip():
        return ToolResult.failure("web_search called with an empty query")

    if offline:
        return _offline_search(query, max_results)

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # older package name, still works
    except ImportError:
        return ToolResult.failure(
            "no search backend installed (pip install -r requirements.txt)"
        )

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:  # network errors, rate limits, etc.
        return ToolResult.failure(f"web_search failed for query {query!r}: {e}")

    results = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
        if r.get("href")
    ]
    if not results:
        return ToolResult.success([], empty=True)
    return ToolResult.success(results)


def _offline_search(query: str, max_results: int) -> ToolResult:
    index_path = FIXTURES_DIR / "search_index.json"
    if not index_path.exists():
        return ToolResult.failure(f"offline fixture index not found at {index_path}")

    index = json.loads(index_path.read_text())
    q_words = set(re.findall(r"[a-z0-9]+", query.lower()))

    scored = []
    for topic in index["topics"]:
        overlap = len(q_words & set(topic["keywords"]))
        if overlap:
            scored.append((overlap, topic))
    scored.sort(key=lambda t: -t[0])

    if not scored:
        return ToolResult.success([], empty=True)

    results = scored[0][1]["results"][:max_results]
    return ToolResult.success(results)


# --------------------------------------------------------------------------
# fetch_page
# --------------------------------------------------------------------------

def fetch_page(url: str, offline: bool = False, timeout: int = 15) -> ToolResult:
    if not url or not url.strip():
        return ToolResult.failure("fetch_page called with an empty url")

    if offline:
        return _offline_fetch(url)

    try:
        import requests
        import trafilatura
    except ImportError as e:
        return ToolResult.failure(f"missing dependency for fetch_page: {e}")

    try:
        resp = requests.get(
            url, timeout=timeout, headers={"User-Agent": "research-agent/1.0"}
        )
        resp.raise_for_status()
    except Exception as e:
        return ToolResult.failure(f"fetch_page failed for {url}: {e}")

    text = trafilatura.extract(resp.text) or ""
    text = text.strip()
    if not text:
        return ToolResult.success({"url": url, "text": ""}, empty=True)

    # Keep pages from blowing up the context window.
    MAX_CHARS = 8000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n...[truncated]"

    return ToolResult.success({"url": url, "text": text})


def _offline_fetch(url: str) -> ToolResult:
    page_path = FIXTURES_DIR / "pages" / f"{_url_slug(url)}.txt"
    if not page_path.exists():
        # Simulate the "tool returns nothing useful" case for one specific
        # fixture URL so the graceful-failure path is actually exercised.
        return ToolResult.success({"url": url, "text": ""}, empty=True)
    text = page_path.read_text().strip()
    return ToolResult.success({"url": url, "text": text})


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------

def summarise(
    text: str,
    question: str,
    offline: bool = False,
    client: Any = None,
    model: str = "claude-sonnet-4-6",
) -> ToolResult:
    if not text or not text.strip():
        return ToolResult.failure("summarise called with empty text")

    if offline or client is None:
        return _offline_summarise(text, question)

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=250,
            messages=[{
                "role": "user",
                "content": (
                    "Summarise the following page in 2-3 sentences, focused ONLY on "
                    f"what is relevant to answering this research question:\n\n"
                    f"Question: {question}\n\nPage text:\n{text[:6000]}\n\n"
                    "Be concise and factual. If the page has nothing relevant, say so."
                ),
            }],
        )
        note = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    except Exception as e:
        return ToolResult.failure(f"summarise (LLM) failed: {e}")

    if not note:
        return ToolResult.success("", empty=True)
    return ToolResult.success(note)


def _offline_summarise(text: str, question: str) -> ToolResult:
    """Deterministic extractive fallback -- no LLM call, used in offline demo mode."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    q_words = set(re.findall(r"[a-z0-9]+", question.lower())) - {
        "what", "when", "who", "where", "how", "is", "are", "the", "a", "of", "was"
    }

    def score(s: str) -> int:
        s_words = set(re.findall(r"[a-z0-9]+", s.lower()))
        return len(q_words & s_words)

    ranked = sorted(sentences, key=score, reverse=True)
    top = [s for s in ranked[:2] if s.strip()]
    if not top:
        return ToolResult.success("", empty=True)
    return ToolResult.success(" ".join(top))
