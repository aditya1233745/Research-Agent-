"""
Shared state that flows through every node of the LangGraph graph.

Kept as a plain TypedDict (rather than a langchain object) so the graph
has zero dependency on the langchain ecosystem beyond langgraph itself.
"""
from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional


class Source(TypedDict):
    id: str              # e.g. "S1" -- the citation tag used in the final answer
    url: str
    title: str
    snippet: str          # short blurb from search results, before fetching
    fetched: bool          # whether fetch_page has been run on this source
    text: str              # full extracted text, once fetched (may be truncated)
    note: str              # the summary produced for this source, once summarised


class AgentState(TypedDict):
    question: str
    max_steps: int
    step: int
    done: bool

    # Running transcript of (role, content) pairs shown to the planner LLM.
    # role is one of: "system", "user", "assistant", "tool"
    messages: List[Dict[str, str]]

    # url -> Source. The single source of truth for anything the agent has found.
    sources: Dict[str, Source]

    # Ordered list of source ids that have been summarised -- i.e. are eligible
    # to be cited in the final answer.
    notes: List[str]

    # Non-fatal problems encountered along the way (failed fetch, empty search, etc).
    # Surfaced in the final trace/README so failures are visible, not silently eaten.
    errors: List[str]

    # Full log of every planner decision + tool result, used to render the
    # human-readable transcript (examples/run_*.md).
    trace: List[Dict[str, Any]]

    answer: Optional[str]

    # Internal: the planner's most recent decision, consumed by the router
    # and the tool node it routes to. Not meant to be read outside graph.py.
    _pending_action: Optional[Dict[str, Any]]
