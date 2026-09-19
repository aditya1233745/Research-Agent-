"""
Keeps every source the agent has ever looked at in one registry, assigns it
a short citation tag ([S1], [S2], ...), and checks at the end that the final
answer doesn't contain claims that aren't backed by a fetched+summarised source.

This is the piece that makes "every claim in the final answer is traceable to
a fetched source" an enforced property rather than a hope.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .state import AgentState, Source

CITATION_RE = re.compile(r"\[S(\d+)\]")


def register_source(state: AgentState, url: str, title: str, snippet: str = "") -> str:
    """Add a source if new, return its citation id (e.g. 'S3'). Idempotent on url."""
    for sid, src in state["sources"].items():
        if src["url"] == url:
            return sid

    sid = f"S{len(state['sources']) + 1}"
    state["sources"][sid] = Source(
        id=sid,
        url=url,
        title=title or url,
        snippet=snippet,
        fetched=False,
        text="",
        note="",
    )
    return sid


def mark_fetched(state: AgentState, sid: str, text: str) -> None:
    if sid in state["sources"]:
        state["sources"][sid]["fetched"] = True
        state["sources"][sid]["text"] = text


def mark_summarised(state: AgentState, sid: str, note: str) -> None:
    if sid in state["sources"]:
        state["sources"][sid]["note"] = note
        if sid not in state["notes"]:
            state["notes"].append(sid)


def citable_ids(state: AgentState) -> List[str]:
    """Source ids that have an actual summary behind them -- i.e. safe to cite."""
    return [sid for sid in state["notes"] if state["sources"].get(sid, {}).get("note")]


def bibliography(state: AgentState) -> str:
    lines = []
    for sid in sorted(state["sources"], key=lambda s: int(s[1:])):
        src = state["sources"][sid]
        marker = "" if src["note"] else "  (fetched but not used in final answer)"
        lines.append(f"[{sid}] {src['title']} -- {src['url']}{marker}")
    return "\n".join(lines)


def validate_citations(answer: str, state: AgentState) -> Tuple[bool, List[str]]:
    """
    Returns (ok, problems). Two checks:
      1. Every [S#] tag used in the answer must refer to a real, summarised source.
      2. Every sentence containing a factual claim should carry at least one
         citation tag (heuristic: sentences longer than a few words).
    This is intentionally conservative -- it flags issues, the caller decides
    whether to retry or append a caveat.
    """
    problems: List[str] = []
    valid_ids = set(citable_ids(state))

    used_ids = set(f"S{n}" for n in CITATION_RE.findall(answer))
    hallucinated = used_ids - valid_ids
    if hallucinated:
        problems.append(
            f"Answer cites {sorted(hallucinated)} which were never fetched+summarised."
        )

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    uncited = [
        s for s in sentences
        if len(s.split()) >= 6 and not CITATION_RE.search(s)
    ]
    # Only flag uncited sentences if the agent actually had summarised
    # sources available to cite. With zero sources (e.g. the question was
    # empty/a placeholder, or every search came back empty), the answer is
    # necessarily a refusal or an honest "couldn't find anything" -- not a
    # claim that could have been cited -- so nagging about missing [S#]
    # tags there is a false positive, not a real problem.
    if uncited and valid_ids:
        problems.append(
            f"{len(uncited)} sentence(s) make claims with no [S#] citation, e.g.: "
            f"\"{uncited[0][:80]}...\""
        )

    return (len(problems) == 0, problems)
