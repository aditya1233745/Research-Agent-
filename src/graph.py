"""
The LangGraph state machine.

    START -> plan -> (search | fetch | summarise | finish) -> plan -> ... -> finish -> END

"plan" is the only node that talks to the planner (LLM or scripted). Every
other node is a thin, deterministic wrapper around one tool call. The hard
step budget is enforced in `route_after_plan`, not just suggested in the
prompt -- if step >= max_steps, the router forces "finish" no matter what
the planner asked for, so the agent physically cannot loop forever.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, END

from . import tools
from .citations import (
    bibliography,
    mark_fetched,
    mark_summarised,
    register_source,
    validate_citations,
)
from .state import AgentState


def build_graph(planner: Any, offline: bool, llm_client: Any = None, model: str = "claude-sonnet-4-6"):
    def log(state: AgentState, kind: str, payload: Dict[str, Any]) -> None:
        state["trace"].append({"step": state["step"], "kind": kind, **payload})

    # ---- plan --------------------------------------------------------

    def plan_node(state: AgentState) -> AgentState:
        state["step"] += 1
        decision = planner.decide(state)
        log(state, "plan", {"decision": decision})
        state["_pending_action"] = decision  # type: ignore[typeddict-item]
        return state

    def route_after_plan(state: AgentState) -> str:
        decision = state["_pending_action"]  # type: ignore[typeddict-item]
        action = decision.get("action", "finish")

        if state["step"] >= state["max_steps"] and action != "finish":
            state["errors"].append(
                f"Step budget ({state['max_steps']}) reached -- forcing finish."
            )
            log(state, "budget", {"forced_finish_at_step": state["step"]})
            return "finish"

        if action not in {"search", "fetch", "summarise", "finish"}:
            state["errors"].append(f"Planner requested unknown action {action!r} -- finishing.")
            return "finish"
        return action

    # ---- search --------------------------------------------------------

    def search_node(state: AgentState) -> AgentState:
        args = state["_pending_action"]["args"]  # type: ignore[typeddict-item]
        query = args.get("query", state["question"])
        result = tools.web_search(query, offline=offline)

        if not result.ok:
            state["errors"].append(result.error or "web_search failed")
            log(state, "tool_error", {"tool": "search", "query": query, "error": result.error})
            return state

        if result.empty:
            state["errors"].append(f"web_search for {query!r} returned no results")
            log(state, "tool_empty", {"tool": "search", "query": query})
            return state

        new_ids = []
        for r in result.data:
            sid = register_source(state, r["url"], r.get("title", ""), r.get("snippet", ""))
            new_ids.append(sid)
        log(state, "tool_result", {"tool": "search", "query": query, "source_ids": new_ids})
        return state

    # ---- fetch --------------------------------------------------------

    def fetch_node(state: AgentState) -> AgentState:
        args = state["_pending_action"]["args"]  # type: ignore[typeddict-item]
        sid = args.get("source_id")
        src = state["sources"].get(sid)

        if not src:
            state["errors"].append(f"fetch requested unknown source_id {sid!r}")
            log(state, "tool_error", {"tool": "fetch", "source_id": sid, "error": "unknown source_id"})
            return state

        result = tools.fetch_page(src["url"], offline=offline)

        if not result.ok:
            state["errors"].append(f"fetch_page failed for [{sid}] {src['url']}: {result.error}")
            log(state, "tool_error", {"tool": "fetch", "source_id": sid, "error": result.error})
            return state

        text = result.data.get("text", "")
        mark_fetched(state, sid, text)

        if result.empty or not text:
            state["errors"].append(f"fetch_page for [{sid}] {src['url']} returned no usable text")
            log(state, "tool_empty", {"tool": "fetch", "source_id": sid})
            return state

        log(state, "tool_result", {"tool": "fetch", "source_id": sid, "chars": len(text)})
        return state

    # ---- summarise --------------------------------------------------------

    def summarise_node(state: AgentState) -> AgentState:
        args = state["_pending_action"]["args"]  # type: ignore[typeddict-item]
        sid = args.get("source_id")
        src = state["sources"].get(sid)

        if not src or not src["fetched"]:
            state["errors"].append(f"summarise requested for [{sid}] which hasn't been fetched")
            log(state, "tool_error", {"tool": "summarise", "source_id": sid, "error": "not fetched"})
            return state

        result = tools.summarise(
            src["text"], state["question"], offline=offline, client=llm_client, model=model
        )

        if not result.ok:
            state["errors"].append(f"summarise failed for [{sid}]: {result.error}")
            log(state, "tool_error", {"tool": "summarise", "source_id": sid, "error": result.error})
            return state

        if result.empty or not result.data:
            state["errors"].append(f"summarise for [{sid}] produced nothing usable")
            log(state, "tool_empty", {"tool": "summarise", "source_id": sid})
            return state

        mark_summarised(state, sid, result.data)
        log(state, "tool_result", {"tool": "summarise", "source_id": sid, "note": result.data})
        return state

    # ---- finish --------------------------------------------------------

    def finish_node(state: AgentState) -> AgentState:
        decision = state["_pending_action"]  # type: ignore[typeddict-item]
        answer = decision.get("args", {}).get("answer", "") or ""

        if not answer:
            answer = "I was unable to produce a cited answer within the step budget."

        ok, problems = validate_citations(answer, state)

        if not ok and state["step"] < state["max_steps"]:
            # Enforce the citation requirement rather than just flagging it:
            # send the agent back to "plan" with the specific problems as an
            # error, so it gets a real chance to fix the answer (e.g. add
            # missing [S#] tags) instead of shipping an uncited answer with
            # a warning note bolted on. Only falls through to accepting a
            # flagged answer once the step budget is actually exhausted.
            state["errors"].append(
                "Your last finish attempt was rejected: " + "; ".join(problems)
                + " Revise the answer so every factual sentence ends with a "
                "real [S#] tag, using only source ids you've summarised, then finish again."
            )
            log(state, "citation_check_failed_retry", {"problems": problems})
            return state

        if not ok:
            state["errors"].extend(problems)
            answer += (
                "\n\n[Note: citation check flagged issues with this answer -- "
                + "; ".join(problems) + "]"
            )
            log(state, "citation_check_failed", {"problems": problems})
        else:
            log(state, "citation_check_passed", {})

        answer += "\n\nSources:\n" + bibliography(state)
        state["answer"] = answer
        state["done"] = True
        return state

    def route_after_finish(state: AgentState) -> str:
        return "finish_end" if state["done"] else "plan"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("summarise", summarise_node)
    graph.add_node("finish", finish_node)

    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"search": "search", "fetch": "fetch", "summarise": "summarise", "finish": "finish"},
    )
    graph.add_edge("search", "plan")
    graph.add_edge("fetch", "plan")
    graph.add_edge("summarise", "plan")
    graph.add_conditional_edges(
        "finish", route_after_finish, {"plan": "plan", "finish_end": END}
    )

    return graph.compile()
