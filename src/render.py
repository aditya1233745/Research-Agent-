"""Turns a finished AgentState's trace into a readable markdown transcript."""
from __future__ import annotations

from .state import AgentState


def render_markdown(state: AgentState, mode_label: str) -> str:
    lines = [
        f"# Run: {state['question']}",
        "",
        f"*Mode: {mode_label} | Step budget: {state['max_steps']} | Steps used: {state['step']}*",
        "",
        "## Trace",
        "",
    ]

    for entry in state["trace"]:
        step = entry["step"]
        kind = entry["kind"]

        if kind == "plan":
            d = entry["decision"]
            lines.append(f"**Step {step} -- plan:** `{d.get('action')}` -- {d.get('reasoning', '')}")
        elif kind == "tool_result":
            tool = entry["tool"]
            if tool == "search":
                lines.append(
                    f"  -> search(`{entry['query']}`) found sources: {', '.join(entry['source_ids'])}"
                )
            elif tool == "fetch":
                lines.append(f"  -> fetch(`{entry['source_id']}`) got {entry['chars']} chars")
            elif tool == "summarise":
                lines.append(f"  -> summarise(`{entry['source_id']}`): {entry['note']}")
        elif kind == "tool_empty":
            lines.append(f"  -> **{entry['tool']}** returned nothing useful (handled gracefully)")
        elif kind == "tool_error":
            lines.append(f"  -> **{entry['tool']}** failed: {entry.get('error')} (handled gracefully)")
        elif kind == "budget":
            lines.append(f"  -> **step budget reached** ({entry['forced_finish_at_step']}) -- forcing finish")
        elif kind == "citation_check_passed":
            lines.append("  -> citation check: all claims traceable to sources ✅")
        elif kind == "citation_check_failed":
            lines.append(f"  -> citation check: **issues found** -- {entry['problems']}")
        elif kind == "citation_check_failed_retry":
            lines.append(
                f"  -> citation check: **issues found, sent back to revise** -- {entry['problems']}"
            )

    lines += ["", "## Final Answer", "", state["answer"] or "(no answer produced)"]

    if state["errors"]:
        lines += ["", "## Errors / non-fatal issues encountered", ""]
        lines += [f"- {e}" for e in state["errors"]]

    return "\n".join(lines)
