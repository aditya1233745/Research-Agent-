"""
The "brain" of the agent: given the current state (question, sources gathered
so far, notes, errors), decide what to do next.

Two implementations behind the same interface:
  - ClaudePlanner   : asks Claude for a JSON action. Used in real runs.
  - ScriptedPlanner : deterministic, no LLM/network. Used in --offline demo
                      mode so the graph, step budget and citation logic can
                      be exercised end-to-end without an API key.

Action schema (both planners must produce this shape):
    {
      "action": "search" | "fetch" | "summarise" | "finish",
      "args": {...},        # action-specific, see graph.py
      "reasoning": "..."    # one line, shown in the trace
    }
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .citations import citable_ids
from .state import AgentState

SYSTEM_PROMPT = """You are a careful research agent. You answer a question by \
calling tools: search the web, fetch a specific page, summarise a fetched page, \
or finish with a final answer.

Rules:
- You have a hard step budget. Use it efficiently: search first, then fetch the \
most promising 2-3 results, then summarise each fetched page, then finish.
- Only "finish" once you have at least one summarised source (unless every \
search/fetch attempt has failed, in which case finish and say so honestly).
- Every sentence in your final answer that states a fact MUST end with a \
citation tag like [S1] referring to a source id you were given. Never invent \
a source id. Never cite a source that hasn't been summarised yet.
- If a tool failed or returned nothing, don't repeat the exact same call -- \
try a different query or a different result instead.
- Write the final answer as plain prose sentences (no markdown headers, bold, \
or numbered/bulleted lists) so each sentence can carry its own [S#] tag \
cleanly. Keep it to roughly 4-8 sentences -- thorough but not exhaustive.

Example of a correctly formatted finish action (note: plain sentences, no \
headers or lists, every factual sentence ends with its citation before the \
period):
{"action": "finish", "args": {"answer": "The housing bubble was fueled by \
widespread subprime lending with little income verification [S1]. Banks \
repackaged these mortgages into mortgage-backed securities that received \
misleadingly high credit ratings [S2]. When home prices fell and defaults \
rose, the value of these securities collapsed, freezing credit markets \
worldwide [S1]."}, "reasoning": "Enough summarised sources to answer."}

Respond with ONLY a JSON object, no prose, no markdown fences:
{"action": "search|fetch|summarise|finish", "args": {...}, "reasoning": "one short sentence"}

Args by action:
  search:    {"query": "..."}
  fetch:     {"source_id": "S1"}
  summarise: {"source_id": "S1"}
  finish:    {"answer": "final answer text with [S#] citations"}
"""


def _render_state_for_llm(state: AgentState) -> str:
    lines = [f"Question: {state['question']}", f"Step {state['step']}/{state['max_steps']}", ""]

    if state["sources"]:
        lines.append("Known sources:")
        for sid, src in state["sources"].items():
            flags = []
            flags.append("fetched" if src["fetched"] else "not fetched")
            flags.append("summarised" if src["note"] else "not summarised")
            lines.append(f"  [{sid}] {src['title']} ({', '.join(flags)}) -- {src['url']}")
            if src["snippet"] and not src["fetched"]:
                lines.append(f"      snippet: {src['snippet'][:150]}")
            if src["note"]:
                lines.append(f"      note: {src['note']}")
    else:
        lines.append("No sources found yet.")

    if state["errors"]:
        lines.append("\nRecent errors:")
        for e in state["errors"][-3:]:
            lines.append(f"  - {e}")

    citable = citable_ids(state)
    lines.append(f"\nSource ids you may cite right now: {citable if citable else '(none yet)'}")
    return "\n".join(lines)


class ClaudePlanner:
    def __init__(self, client: Any, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def decide(self, state: AgentState) -> Dict[str, Any]:
        user_content = _render_state_for_llm(state)
        try:
            msg = self._create(user_content)
            raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        except Exception as e:
            return {
                "action": "finish",
                "args": {"answer": ""},
                "reasoning": f"planner LLM call failed ({e}); forced finish",
            }

        action = _parse_json_action(raw)
        if action is None:
            salvaged = _salvage_truncated_finish(raw)
            if salvaged is not None:
                return salvaged
            return {
                "action": "finish",
                "args": {"answer": ""},
                "reasoning": f"planner returned unparseable output: {raw[:200]!r}",
            }
        return action

    def _create(self, user_content: str) -> Any:
        """
        Calls messages.create with temperature=0.2 (more consistent
        instruction-following than the SDK default). Falls back to omitting
        temperature if the installed anthropic SDK version doesn't accept
        it, rather than hard-failing every single call over one kwarg.
        """
        kwargs = dict(
            model=self.model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        try:
            return self.client.messages.create(temperature=0.2, **kwargs)
        except TypeError as e:
            if "temperature" not in str(e):
                raise
            return self.client.messages.create(**kwargs)


def _parse_json_action(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    # tolerate accidental markdown fences
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    obj = _try_json(raw)
    if obj is None:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            obj = _try_json(match.group(0))

    if obj is None or "action" not in obj:
        return None
    obj.setdefault("args", {})
    obj.setdefault("reasoning", "")
    return obj


def _try_json(candidate: str) -> Optional[Dict[str, Any]]:
    """
    json.loads with one repair pass for a real failure mode seen from Claude:
    it sometimes writes \\' (backslash-escaped single quote) inside a JSON
    string value -- valid in Python/JS string literals, but \\' is not a
    legal JSON escape sequence, so the standard parser rejects the whole
    object. \\' is never meaningful in JSON (single quotes never need
    escaping there), so it's always safe to unescape it before parsing.
    """
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(candidate.replace("\\'", "'"))
    except json.JSONDecodeError:
        return None


_TRUNCATED_FINISH_RE = re.compile(
    r'"action"\s*:\s*"finish".*?"answer"\s*:\s*"(?P<answer>.*)', re.DOTALL
)


def _salvage_truncated_finish(raw: str) -> Optional[Dict[str, Any]]:
    """
    Last-resort recovery for a response that got cut off by max_tokens mid
    -answer (hit in practice: a long, well-cited answer with several [S#]
    tags ran past the token budget, so the JSON never got its closing quote
    /brace -- the whole action was being thrown away even though the model
    had done all the real work). If the raw text clearly started writing a
    "finish" action with an "answer" field, pull out whatever prose made it
    through before the cutoff rather than discarding it outright.
    """
    match = _TRUNCATED_FINISH_RE.search(raw)
    if not match:
        return None

    partial = match.group("answer")
    # undo JSON string escaping on what we did get, best-effort
    partial = partial.replace('\\"', '"').replace("\\n", " ").replace("\\'", "'")
    # trim a dangling half-written sentence/citation at the very end
    partial = re.sub(r"[^.!?\]]*$", "", partial).strip()
    if not partial:
        return None

    return {
        "action": "finish",
        "args": {"answer": partial},
        "reasoning": "planner's response was cut off mid-answer; recovered the complete sentences before the cutoff",
    }


class ScriptedPlanner:
    """
    Deterministic stand-in for the LLM, used in --offline mode.

    Follows the exact same policy Claude is prompted to follow: search once,
    fetch up to 3 results, summarise each fetched-but-unsummarised source,
    then finish. This lets the graph, step budget, tool-failure handling and
    citation validation all be exercised without any network or API key.
    """

    def decide(self, state: AgentState) -> Dict[str, Any]:
        sources = state["sources"]

        if not sources:
            return {
                "action": "search",
                "args": {"query": state["question"]},
                "reasoning": "No sources yet -- start with a web search.",
            }

        unfetched = [sid for sid, s in sources.items() if not s["fetched"]]
        fetched_unsummarised = [
            sid for sid, s in sources.items() if s["fetched"] and not s["note"] and s["text"]
        ]
        fetched_but_empty = [
            sid for sid, s in sources.items() if s["fetched"] and not s["text"]
        ]

        if fetched_unsummarised:
            sid = fetched_unsummarised[0]
            return {
                "action": "summarise",
                "args": {"source_id": sid},
                "reasoning": f"[{sid}] was fetched and has text -- summarise it next.",
            }

        # Fetch up to 3 sources total before giving up and moving to finish.
        fetch_attempts = sum(1 for s in sources.values() if s["fetched"])
        if unfetched and fetch_attempts < 3:
            sid = unfetched[0]
            return {
                "action": "fetch",
                "args": {"source_id": sid},
                "reasoning": f"Fetch [{sid}] to get its full text.",
            }

        citable = citable_ids(state)
        if citable:
            # Tag every sentence of every note with its own source, rather than
            # bunching all citations at the end -- this is what the citation
            # validator (src/citations.py) actually checks for.
            import re as _re

            parts = []
            for sid in citable:
                note = sources[sid]["note"]
                for sentence in _re.split(r"(?<=[.!?])\s+", note.strip()):
                    if sentence:
                        parts.append(f"{sentence.rstrip('.')} [{sid}].")
            answer = " ".join(parts)
            return {
                "action": "finish",
                "args": {"answer": answer},
                "reasoning": "Have at least one summarised source -- finish with a per-sentence-cited answer.",
            }

        reason = "no source could be fetched or summarised" if fetched_but_empty else "ran out of leads"
        return {
            "action": "finish",
            "args": {"answer": f"I could not find a reliable answer ({reason})."},
            "reasoning": "No citable sources available -- finish honestly rather than guess.",
        }
