# supervisor-agent

A LangGraph multi-agent app: a supervisor orchestrating a researcher and a copywriter agent. Built incrementally while following [ai-launchpad](https://github.com/kenneth-liao/ai-launchpad).

## Setup

```bash
uv sync
cp .env.example .env  # then fill in GROQ_API_KEY and TAVILY_API_KEY (both have free tiers)
```

## Structure

```
supervisor_agent/
  main.py         # entry point
  supervisor.py   # supervisor graph/orchestration
  researcher.py   # researcher agent
  copywriter.py   # copywriter agent
  prompts/        # agent system prompts
```