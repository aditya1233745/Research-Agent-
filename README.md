# Tool-Using Research Agent

A small agent that answers a research question by deciding, step by step, which
tool to call next — search the web, fetch a page, or summarise it — and stops
itself with a hard step budget. Every sentence in its final answer carries a
`[S#]` tag pointing at a source it actually fetched and summarised.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) as an explicit
state machine, [Claude](https://www.anthropic.com/claude) as the planner, and
DuckDuckGo for search.

## How it works

```
START -> plan -> (search | fetch | summarise | finish) -> plan -> ... -> finish -> END
```

- **`plan`** is the only node that thinks. It looks at the question, the
  sources found so far (fetched? summarised?), and any recent errors, and
  decides the single next action as JSON: `search`, `fetch`, `summarise`, or
  `finish`.
- **`search` / `fetch` / `summarise`** are thin, deterministic wrappers around
  one tool call each. They never call the LLM.
- **`finish`** assembles the final answer, runs a citation check against it,
  and appends a bibliography.

### The four things this was built to satisfy

| Requirement | Where it lives |
|---|---|
| At least two distinct tools | `src/tools.py`: `web_search`, `fetch_page`, `summarise` (three) |
| A hard step limit | `src/graph.py::route_after_plan` forces `finish` once `step >= max_steps`, **regardless of what the planner asks for** — this is a code-level cutoff, not just a prompt instruction |
| Every claim traceable to a fetched source | `src/citations.py::validate_citations` — every `[S#]` in the answer must reference a source that was actually fetched *and* summarised; every claim-length sentence must carry a tag. **This is enforced, not just flagged**: `src/graph.py::finish_node` rejects an uncited answer and routes the agent back to `plan` to try again (with the specific problem in `state["errors"]` so the planner sees exactly what to fix), as long as step budget remains. Only once the budget is exhausted does an uncited answer get shipped anyway, clearly marked with a warning note. |
| Graceful handling of tool failure / empty results | Every tool returns a `ToolResult(ok, data, error, empty)` instead of raising. `graph.py` logs the failure, adds it to `state["errors"]`, and lets the planner route around it (e.g. try a different source) instead of crashing |

### Step budget in practice

The budget is enforced in two places on purpose:
1. The planner's system prompt tells it the budget, so it *tries* to wrap up in time.
2. `route_after_plan` in `graph.py` checks `state["step"] >= state["max_steps"]`
   and forces the `finish` node no matter what action the planner requested.
   Point 2 is the one that actually matters — a buggy or adversarial planner
   response can't make the graph loop forever.

## Project layout

```
src/
  state.py      AgentState TypedDict — the single object that flows through the graph
  tools.py      web_search / fetch_page / summarise, each with a real + offline path
  citations.py  source registry, citation-tag validation, bibliography rendering
  planner.py    ClaudePlanner (real) and ScriptedPlanner (offline demo, no LLM)
  graph.py      the LangGraph StateGraph wiring plan + tool nodes together
  render.py     turns a finished run into a markdown transcript
  agent.py      CLI entrypoint
fixtures/       offline demo data (search index + fetched pages) — see below
examples/       3 saved run transcripts (see below)
tests/          pytest covering all four assessed properties, run offline
```

## Setup

```bash
git clone <this-repo>
cd research-agent
pip install -r requirements.txt
cp .env.example .env    # then put your ANTHROPIC_API_KEY in .env
```

## Running it

**Live** (real DuckDuckGo search + real Claude reasoning, needs an API key):

```bash
python -m src.agent --question "What caused the 2008 financial crisis?" --max-steps 8
```

**Offline demo** (no network, no API key — runs the exact same graph and tools,
but with a scripted planner and canned fixture pages instead of live calls):

```bash
python -m src.agent --offline --question "What year was the Eiffel Tower completed?"
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--question` / `-q` | required | the research question |
| `--max-steps` | `8` | hard step budget |
| `--offline` | off | use the scripted planner + fixtures instead of live Claude/DuckDuckGo |
| `--model` | `claude-sonnet-4-6` | override the Claude model |
| `--save-run PATH` | — | also write the markdown transcript to a file |

## Why an offline mode exists

The graph, the step budget, the citation validator, and the tool-failure
handling are all pure control-flow logic that has nothing to do with which
LLM or search engine is behind them. `--offline` swaps in a
`ScriptedPlanner` (a deterministic stand-in that follows the same
search-then-fetch-then-summarise-then-finish policy the real planner is
prompted to follow) and fixture data in `fixtures/`, so the whole thing can
be exercised — and unit tested — with **zero network access and zero API
key**. The 3 example runs below and the test suite (`tests/test_agent.py`)
both use this mode so they're reproducible by anyone who clones the repo.

The fixtures were deliberately built to include a couple of dead links with
no fixture file behind them, specifically to exercise the "tool returns
nothing" path — see run 1 and run 2 below, source `S3` in both.

Live mode uses the same graph unchanged, just with `ClaudePlanner` +
real DuckDuckGo/`requests`+`trafilatura` behind the tool calls.

## Example runs

All three below were generated by actually running `python -m src.agent
--offline ...` — nothing here is hand-written. Full transcripts are in
[`examples/`](./examples).

### 1. Normal success path — [full transcript](./examples/run_1_eiffel_tower.md)

`--question "What year was the Eiffel Tower completed and who built it?"`

Search returns 3 sources. The agent fetches and summarises the first two,
attempts to fetch the third (`S3`, a deliberately dead fixture link), gets
nothing back, logs it as a handled error, and moves on rather than getting
stuck. Finishes in 7 of 8 steps with a fully-cited answer:

> It was designed and built by the company of civil engineer Gustave Eiffel
> for the 1889 World's Fair... **[S1]**. ... Gustave Eiffel (1832-1923) was a
> French civil engineer **[S2]**.

Citation check: ✅ all claims traceable to `S1`/`S2`. `S3` shows up in the
bibliography, honestly marked as fetched-but-unused.

### 2. Another success path, different topic — [full transcript](./examples/run_2_great_wall.md)

`--question "How long is the Great Wall of China and when was it built?"`

Same shape as run 1 — 3 search results, 2 successfully fetched and
summarised, 1 dead link (`S3`) hit and gracefully skipped. Demonstrates the
pattern is not a one-off.

### 3. Total failure, handled honestly — [full transcript](./examples/run_3_unanswerable.md)

`--question "What is the capital of a fictional planet called Zorblax-9?"` with `--max-steps 5`

Search returns nothing relevant on every attempt (the fixture index has no
match — in live mode this models a query with no useful results). Every
`web_search` failure is logged to `state["errors"]` instead of crashing the
run. Once the step budget is hit, `route_after_plan` **forces** the `finish`
node regardless of what the planner would have preferred:

> I was unable to produce a cited answer within the step budget.

No hallucinated citation, no infinite retry loop, no crash — just an honest
"I don't know" plus the full list of what was tried and why it didn't work.
(The citation checker correctly does *not* flag this answer: with zero
sources ever found, there's nothing to cite, and refusing honestly isn't a
claim that needs backing — see the note on this in "Known limitations".)

## Live-mode bugs found and fixed while testing on Windows

Running this against the real Claude API + real DuckDuckGo surfaced three
issues the offline demo mode couldn't catch, since it never calls an LLM.
All three are now covered by regression tests in `tests/test_agent.py`:

1. **Invalid JSON escape from Claude.** Claude occasionally writes `\'`
   (backslash-escaped apostrophe) inside a JSON string value — valid in
   Python/JS, but not legal JSON, so the strict parser rejected the whole
   action. Fixed with a repair pass in `planner.py::_try_json`.
2. **Truncated answers.** A long, well-cited answer sometimes ran past the
   response token budget, so the JSON never closed and the entire action —
   including real, correct work — was discarded. Fixed by raising the token
   budget, instructing the model to answer in plain cited prose instead of
   markdown (shorter, more parseable), and adding a last-resort
   `_salvage_truncated_finish` that recovers whatever complete, cited
   sentences made it through before the cutoff.
3. **Uncited answers slipping through.** The model doesn't reliably follow
   a "cite every sentence" instruction just because it's asked to. The
   citation check now actually **rejects** an uncited `finish` attempt and
   sends the agent back to revise it (see the table above), rather than
   shipping the bad answer with a warning note.
4. *(Not a bug, but a papercut)* the `duckduckgo-search` PyPI package was
   renamed to `ddgs`; `tools.py` now tries `ddgs` first and falls back to
   the old name. Windows console/file output also needed an explicit
   `encoding="utf-8"` to avoid crashing on non-ASCII page content (cp1252
   is not a great default codepage for the open web).

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Covers, all offline: two distinct tools actually get called in a normal run,
the step budget is never exceeded, the final answer's citations all trace
back to summarised sources, a total search failure is handled without
crashing, the JSON-escape repair, the truncated-answer salvage, and the
reject-and-retry loop for an uncited finish attempt.

## Known limitations

- `ScriptedPlanner` (offline mode) is intentionally simple: it always
  re-issues the exact same query on a failed search rather than trying a
  reformulation. The real `ClaudePlanner` is prompted not to repeat a failed
  call verbatim — see run 3's transcript for how the scripted version
  behaves instead (it still terminates safely via the step budget, just less
  cleverly).
- The citation validator is a heuristic (regex + sentence length), not a
  full entailment check — it catches missing/hallucinated `[S#]` tags, not
  subtly wrong paraphrasing of a source. It also deliberately skips the
  "missing citation" check when the agent found zero usable sources at all
  (e.g. the question was empty/a placeholder, or every search came back
  empty) — in that case the answer is a refusal or an honest "couldn't find
  anything," not a claim, and there's nothing available for it to cite.
- `fetch_page` truncates pages to ~8000 characters before summarising, to
  keep the LLM context small; very long sources may lose detail.
