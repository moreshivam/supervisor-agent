# Supervisor Agent

A multi-agent content research & writing system built with **LangGraph**. Takes a natural language request like *"Write a LinkedIn post about why small businesses should use AI"* and returns a researched, cited draft — a supervisor plans the work, a researcher gathers and reports on it, and a copywriter turns it into the final piece, all automated.

---

## Architecture

```
User Input (natural language request)
        ↓
┌───────────────────────────────────────────────────────┐
│               LANGGRAPH SUPERVISOR GRAPH                │
│                                                         │
│  supervisor   ──► plans 2-4 atomic research angles     │
│       ↓                                                 │
│  researcher     (Tavily search + page extraction)      │
│       ↓           called once per research angle        │
│  supervisor   ──► waits for all research, then...      │
│       ↓                                                 │
│  copywriter     (reads all accumulated research)        │
│       ↓           writes the blog / LinkedIn post        │
│  supervisor   ──► summarizes the result for the user     │
└───────────────────────────────────────────────────────┘
        ↓
CLI  →  streamed [supervisor] / [researcher] / [copywriter] output
```

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| LLM | Groq (Llama 3.3 70B Versatile) | Free |
| Agent Orchestration | LangGraph + `langgraph-supervisor` | Free (open source) |
| Web Search | Tavily Search | Free (1,000 searches/month) |
| Page Extraction | Tavily Extract | Free (same key) |
| Memory / Checkpointing | LangGraph `InMemorySaver` | Free (in-process) |
| Package Management | `uv` | Free (open source) |

**Total cost to run: $0**

---

## Project Structure

```
supervisor-agent/
│
├── .env                             # API keys (never commit this)
├── .env.example                     # blank template, safe to commit
├── .gitignore
├── pyproject.toml                   # uv project + dependencies
│
└── supervisor_agent/
    │
    ├── main.py                      # entry point: CLI runner, retry + error handling
    ├── supervisor.py                # supervisor graph: orchestration, custom handoff tools
    ├── researcher.py                # researcher agent: search + extract + report tools
    ├── copywriter.py                # copywriter agent: reads research, writes content
    ├── state.py                     # shared AgentState schema (messages, reports, content)
    ├── schemas.py                   # Pydantic schemas: ResearchReport, GeneratedContent
    │
    ├── prompts/
    │   ├── supervisor.md            # supervisor's planning + delegation policy
    │   ├── researcher.md            # researcher's tool-use instructions
    │   └── copywriter.md            # copywriter's tool-use instructions
    │
    └── example_content/
        ├── blog.md                  # example blog post, referenced in copywriter's prompt
        └── linkedin.md              # example LinkedIn post, referenced in copywriter's prompt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/moreshivam/supervisor-agent.git
cd supervisor-agent
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Get your API keys

| Key | Where to get it | Free tier |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Rate-limited, no card required |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | 1,000 searches/month, no card required |

### 4. Configure `.env`

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

---

## Running

### CLI

```bash
uv run python -m supervisor_agent.main
```

```
Supervisor agent ready. Type a request (or 'exit' to quit).

You: Write a LinkedIn post about why small businesses should use AI

[supervisor] Research the benefits of AI for small businesses, including increased
efficiency, improved customer service, and enhanced competitiveness. Focus on recent
studies, statistics, and expert opinions.

[researcher] Research report on 'AI benefits for small businesses' saved.

[copywriter] LinkedIn post saved.

[supervisor] Done — I researched AI's impact on small businesses and drafted a
LinkedIn post highlighting the key benefits, backed by cited sources.
```

Type `exit` to quit.

---

## How It Works

### LangGraph State

All three agents share one `AgentState` schema — no agent calls another agent directly, they only read and write shared state:

```python
class AgentState(MessagesState):
    research_reports: Annotated[list[dict], operator.add]   # accumulates, doesn't overwrite
    generated_content: Annotated[list[dict], operator.add]
    remaining_steps: NotRequired[RemainingSteps]
```

Because each agent is itself a compiled LangGraph graph added as a node in the supervisor's graph, LangGraph merges these shared keys back into the parent automatically once that node finishes — no manual plumbing between agents.

### Runtime Agent Routing

The supervisor decides which agent to call, and when, **at runtime** — it's the LLM's own tool-calling decision at each turn, not hardcoded control flow. It's bound to two tools, `handoff_to_researcher` and `handoff_to_copywriter`, each taking an explicit `task_description`. The policy that shapes *how* it uses them (break into 2-4 atomic tasks, call the researcher once per task, wait for all research before calling the copywriter) lives entirely in `prompts/supervisor.md`, in plain language — there's no `if`/`else` router anywhere.

### Parallel Handoff Safety

If the supervisor's LLM ever emits more than one tool call in a single turn, a plain `Command(goto=agent_name)` per call would race — only one `goto` would actually take effect. The handoff tools check for this and fall back to LangGraph's `Send` primitive to fan out to every destination safely within the same superstep, mirroring `langgraph-supervisor`'s own default handoff tool. `research_reports`/`generated_content` also use `operator.add` reducers, so even concurrent writes merge instead of overwriting each other.

### Turn-Level Retry

Groq/Llama occasionally malforms a tool call — a 400 error the SDK's own retry logic won't retry (it only covers 408/409/429/5xx). Instead of crashing or resending the user's message, a failed turn retries by resuming from the graph's last checkpoint (`InMemorySaver`), picking up exactly where it left off.

---

## Key Design Decisions

**Why custom handoff tools instead of `langgraph-supervisor`'s default?**
The default `transfer_to_<agent>` tool just forwards the whole conversation and lets the sub-agent guess its task. The supervisor's plan-then-delegate design needs an explicit, atomic task per call, so custom tools carry a `task_description` argument — built on the library's own supported extension point, not a fork.

**Why one model size for every agent?**
A smaller, faster model (`llama-3.1-8b-instant`) was tried first for the researcher, since it's called several times per request. It reliably failed to format tool calls with large text arguments. Every agent now runs `llama-3.3-70b-versatile` — more reliable, at the cost of burning through the free tier's daily quota faster.

**Why an in-memory checkpointer over Postgres/Redis?**
Zero cost, and enough to support multi-turn conversations and resume-on-failure within a run. A real deployment would swap in a persistent checkpointer behind the same interface.

**Why Groq over OpenAI?**
Groq runs Llama 3.3 70B at free-tier cost with fast inference. No credit card required.

**Why retry at the turn level instead of `.with_retry()`?**
`.with_retry()` wraps the model in a `RunnableRetry` that drops `bind_tools()`, which `create_react_agent` needs — it broke agent construction outright. Retrying the whole turn via checkpoint-resume is the approach that's actually robust to the failure class observed.

---

## What Each Agent Does

| Agent | Reads | Tools Called | Writes |
|---|---|---|---|
| `supervisor` | conversation history | `handoff_to_researcher`, `handoff_to_copywriter` | delegates, then summarizes |
| `researcher` | task description from supervisor | `search_web`, `extract_content_from_webpage`, `generate_research_report` | `research_reports` |
| `copywriter` | `research_reports` | `review_research_reports`, `generate_blog_post`, `generate_linkedin_post` | `generated_content` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Built by [Shivam More](https://github.com/moreshivam)
