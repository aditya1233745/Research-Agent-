"""
Minimal tests covering the four assessment criteria, all run offline
(no network, no API key needed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import run
from src.citations import validate_citations
from src.planner import _parse_json_action, _salvage_truncated_finish


def test_two_distinct_tools_get_used():
    state = run("What year was the Eiffel Tower completed and who built it?", max_steps=8, offline=True, model="n/a")
    tools_used = {e["tool"] for e in state["trace"] if e["kind"] in ("tool_result", "tool_empty", "tool_error")}
    assert {"search", "fetch"}.issubset(tools_used)
    assert len(tools_used) >= 2


def test_step_budget_is_hard():
    state = run("How does photosynthesis work?", max_steps=3, offline=True, model="n/a")
    assert state["step"] <= 3
    assert state["done"] is True


def test_final_answer_citations_are_traceable():
    state = run("How long is the Great Wall of China?", max_steps=8, offline=True, model="n/a")
    ok, problems = validate_citations(state["answer"].split("\n\nSources:")[0], state)
    assert ok, problems


def test_graceful_handling_of_total_search_failure():
    state = run("asdkjasdlkj nonsense query xyz123", max_steps=5, offline=True, model="n/a")
    assert state["done"] is True
    assert any("returned no results" in e for e in state["errors"])
    assert state["answer"] is not None


def test_planner_json_repairs_invalid_backslash_quote():
    # Regression test: seen in a live run where Claude wrote \' (an invalid
    # JSON escape) inside a string value, e.g. "it hasn\'t been fetched".
    raw = (
        '{"action": "fetch", "args": {"source_id": "S1"}, '
        '"reasoning": "it hasn\\\'t been fetched yet"}'
    )
    action = _parse_json_action(raw)
    assert action is not None
    assert action["action"] == "fetch"
    assert action["args"]["source_id"] == "S1"


def test_salvage_recovers_truncated_finish_answer():
    # Regression test: seen in a live run where max_tokens cut the response
    # off mid-answer, so the JSON never closed -- the whole action (and the
    # real work behind it) was being thrown away.
    raw = (
        '{"action": "finish", "args": {"answer": "The crisis was driven by '
        'subprime lending [S1]. Ratings agencies mis-rated the resulting '
        'securities [S2]. When defaults rose, the mark'
    )
    salvaged = _salvage_truncated_finish(raw)
    assert salvaged is not None
    assert salvaged["action"] == "finish"
    assert "[S1]" in salvaged["args"]["answer"]
    assert "[S2]" in salvaged["args"]["answer"]
    # the dangling half-sentence after the last full stop must be dropped
    assert "mark" not in salvaged["args"]["answer"].split("[S2].")[-1]


def test_finish_is_rejected_and_retried_when_uncited():
    """
    An uncited finish attempt (with steps still remaining) must be rejected
    and routed back to the planner instead of silently accepted -- this is
    what makes 'every claim traceable to a source' an enforced property
    rather than a warning label. Uses a tiny scripted fake planner so this
    test needs no network/API key.
    """
    from src.graph import build_graph
    from src.state import AgentState

    class FakePlanner:
        def __init__(self):
            self.calls = 0

        def decide(self, state):
            self.calls += 1
            if self.calls == 1:
                return {"action": "search", "args": {"query": "eiffel tower"}, "reasoning": "start"}
            if self.calls == 2:
                # first two sources from the fixture index get registered by search;
                # fetch+summarise the first one so there's a real citable source
                sid = next(iter(state["sources"]))
                return {"action": "fetch", "args": {"source_id": sid}, "reasoning": "fetch"}
            if self.calls == 3:
                sid = next(iter(state["sources"]))
                return {"action": "summarise", "args": {"source_id": sid}, "reasoning": "summarise"}
            if self.calls == 4:
                # deliberately uncited -- should get rejected
                return {"action": "finish", "args": {"answer": "The tower was built a long time ago by some people."}, "reasoning": "bad finish"}
            # after rejection, try again with a properly cited answer
            sid = next(iter(state["sources"]))
            return {"action": "finish", "args": {"answer": f"It was completed in 1889 [{sid}]."}, "reasoning": "fixed finish"}

    planner = FakePlanner()
    graph = build_graph(planner, offline=True)
    initial_state: AgentState = {
        "question": "When was the Eiffel Tower completed?",
        "max_steps": 8,
        "step": 0,
        "done": False,
        "messages": [],
        "sources": {},
        "notes": [],
        "errors": [],
        "trace": [],
        "answer": None,
        "_pending_action": None,
    }
    final_state = graph.invoke(initial_state, config={"recursion_limit": 30})

    assert final_state["done"] is True
    assert "[S1]" in final_state["answer"]
    assert any("rejected" in e for e in final_state["errors"])
    retry_events = [t for t in final_state["trace"] if t["kind"] == "citation_check_failed_retry"]
    assert len(retry_events) == 1
