# supervisor-agent

A LangGraph multi-agent app: a **supervisor** that plans work and dynamically delegates it to a **researcher** and a **copywriter** agent.

## How it works

A user request goes to the supervisor, which never researches or writes anything itself. It breaks the request into 2-4 atomic research angles, hands each one to the researcher (`handoff_to_researcher(task_description=...)`), waits for all of them to finish, then hands the synthesis off to the copywriter (`handoff_to_copywriter(task_description=...)`). Control always returns to the supervisor between steps, which decides what happens next.

```
User -> Supervisor -> Researcher (x2-4, one call per research angle)
                    -> Copywriter (once, after research is done)
                    -> User (summary)
```

The supervisor decides which agent to call, and when, **at runtime** - it's the LLM's own tool-calling decision at each turn, guided by the policy written in [`prompts/supervisor.md`](supervisor_agent/prompts/supervisor.md), not hardcoded routing logic. See [`supervisor.py`](supervisor_agent/supervisor.py) for the handoff tools that make this possible.

## Setup

```bash
uv sync
cp .env.example .env  # then fill in GROQ_API_KEY and TAVILY_API_KEY (both have free tiers, no card required)
```

Get free API keys at [console.groq.com](https://console.groq.com) and [tavily.com](https://tavily.com).

## Run

```bash
uv run python -m supervisor_agent.main
```

Starts an interactive CLI - type a request, watch the supervisor plan and delegate, type `exit` to quit.

## Project structure

```
supervisor_agent/
  main.py             # entry point: builds the graph, runs an interactive CLI with retry-on-failure
  supervisor.py        # supervisor graph: orchestration, custom handoff tools
  researcher.py         # researcher agent: web search + page extraction + report generation
  copywriter.py          # copywriter agent: reads research, writes blog/linkedin content
  state.py                # shared AgentState schema (messages, research_reports, generated_content)
  schemas.py                # Pydantic schemas for research reports and generated content
  prompts/                    # system prompts for each agent (markdown, loaded at runtime)
  example_content/              # example blog/linkedin posts referenced in the copywriter's prompt
```

## Design notes

- **Zero-cost stack**: Groq (LLM, free tier) + Tavily (search, free tier, 1000 searches/month) - no OpenAI, no paid usage anywhere.
- **Shared state, not message-passing**: all three agents use the same `AgentState` schema. `research_reports` and `generated_content` use `operator.add` reducers so multiple researcher calls accumulate instead of overwriting each other, and so concurrent writes (if the supervisor ever hands off to multiple agents in one turn) merge safely instead of racing.
- **Custom handoff tools**: `langgraph-supervisor`'s default handoff tool just forwards the whole conversation to the next agent. Ours (`_make_handoff_tool` in `supervisor.py`) carries an explicit `task_description` per call, matching the supervisor's plan-then-delegate design, while still using the library's supported extension point rather than forking it. It also mirrors the library's own `Send`-based fan-out for the case where the supervisor hands off to multiple agents in a single turn.
- **Turn-level retry**: Groq/Llama occasionally malforms a tool call (a 400 error the SDK's own retry logic won't retry). `main.py` retries the whole turn by resuming from the last LangGraph checkpoint (`InMemorySaver`) rather than restarting or resending the user's message.
- **Recursion limit**: graph invocation is capped at 25 steps as a backstop against a runaway delegation loop burning through the free-tier quota.

## License

MIT - see [LICENSE](LICENSE).
