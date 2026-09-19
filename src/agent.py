"""
CLI entrypoint.

    python -m src.agent --question "..." [--offline] [--max-steps 8] [--save-run examples/run_1.md]

In real mode, requires ANTHROPIC_API_KEY (loaded from .env). In --offline
mode, no network or API key is needed at all -- see README.md.
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from .graph import build_graph
from .planner import ClaudePlanner, ScriptedPlanner
from .render import render_markdown
from .state import AgentState


def run(question: str, max_steps: int, offline: bool, model: str) -> AgentState:
    llm_client = None
    if offline:
        planner = ScriptedPlanner()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "ANTHROPIC_API_KEY not set. Either put it in a .env file "
                "(see .env.example) or run with --offline for the no-key demo mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        import anthropic

        llm_client = anthropic.Anthropic(api_key=api_key)
        planner = ClaudePlanner(llm_client, model=model)

    graph = build_graph(planner, offline=offline, llm_client=llm_client, model=model)

    initial_state: AgentState = {
        "question": question,
        "max_steps": max_steps,
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

    # generous recursion_limit buffer above our own step budget, since each
    # "step" is (plan node + one tool node) = 2 graph node visits
    final_state = graph.invoke(initial_state, config={"recursion_limit": max_steps * 2 + 10})
    return final_state


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tool-using research agent")
    parser.add_argument("--question", "-q", required=True, help="The research question to answer")
    parser.add_argument("--max-steps", type=int, default=8, help="Hard step budget (default: 8)")
    parser.add_argument("--offline", action="store_true", help="Run with no network/API key, using fixtures")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--save-run", help="Path to save a markdown transcript of this run")
    args = parser.parse_args()

    state = run(args.question, args.max_steps, args.offline, args.model)
    mode_label = "offline demo (scripted planner + fixtures)" if args.offline else f"live ({args.model} + DuckDuckGo)"

    transcript = render_markdown(state, mode_label)
    try:
        print(transcript)
    except UnicodeEncodeError:
        # some Windows terminals default to a narrow codepage (cp1252) that
        # can't print every character a live web page might contain
        print(transcript.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))

    if args.save_run:
        with open(args.save_run, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"\n(transcript saved to {args.save_run})", file=sys.stderr)


if __name__ == "__main__":
    main()
