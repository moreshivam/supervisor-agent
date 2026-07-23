import re
import uuid

from dotenv import load_dotenv

# Must run before importing supervisor_agent.supervisor: that import chain
# instantiates TavilySearch/TavilyExtract at module load time (in
# researcher.py), which read TAVILY_API_KEY immediately.
load_dotenv()

import groq  # noqa: E402

from supervisor_agent.supervisor import build_supervisor_agent  # noqa: E402

# Hard backstop against a confused supervisor looping forever between agents
# (and burning through free-tier Groq quota). Each handoff + agent turn +
# tool call counts as steps, so this allows several research/copywriter
# round trips without being unbounded.
MAX_GRAPH_STEPS = 25

# Groq/Llama tool-calling occasionally malforms a tool call, surfacing as a
# BadRequestError the Groq SDK's own retry logic won't retry (it only covers
# 408/409/429/5xx, not 400s - see researcher.py/copywriter.py/supervisor.py's
# max_retries comments). Retrying here at the turn level, one LLM call away
# from where it failed, works around it.
MAX_TURN_ATTEMPTS = 3


def _format_error(exc: Exception) -> str:
    """Reduce a raw provider/network exception to a short, structured
    message instead of dumping the full error object (status, error code,
    message, and retry-after, with the marketing footer stripped out)."""
    if isinstance(exc, groq.APIStatusError):
        detail = exc.body.get("error", {}) if isinstance(exc.body, dict) else {}
        message = str(detail.get("message", exc)).split("Need more tokens?")[0].strip()

        retry_after = exc.response.headers.get("retry-after")
        if not retry_after:
            match = re.search(r"try again in ([\w.]+)", message)
            retry_after = match.group(1) if match else None

        lines = [f"Groq API error [{exc.status_code} {detail.get('code', type(exc).__name__)}]"]
        lines.append(f"  {message}")
        if retry_after:
            lines.append(f"  retry after: {retry_after}")
        return "\n".join(lines)

    return f"{type(exc).__name__}: {exc}"


def _run_turn(agent, user_input: str, config: dict):
    """Stream one user turn, retrying from the last checkpoint on failure.

    The first attempt sends the new user message. If it raises partway
    through, later attempts pass None as input, which resumes the graph from
    its last saved checkpoint for this thread_id instead of restarting the
    turn or re-appending the user message.
    """
    graph_input = {"messages": [{"role": "user", "content": user_input}]}
    for attempt in range(1, MAX_TURN_ATTEMPTS + 1):
        try:
            for chunk in agent.stream(graph_input, config=config, stream_mode="updates"):
                yield chunk
            return
        except Exception as exc:
            if attempt == MAX_TURN_ATTEMPTS:
                raise
            print(f"\n[retrying after error]\n{_format_error(exc)}")
            graph_input = None


def main():
    agent = build_supervisor_agent()

    # Fixed per-run thread_id: the InMemorySaver checkpointer uses this to
    # keep the conversation's state (including research_reports) across
    # multiple turns in this session.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": MAX_GRAPH_STEPS}

    print("Supervisor agent ready. Type a request (or 'exit' to quit).\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        try:
            for chunk in _run_turn(agent, user_input, config):
                for node_name, update in chunk.items():
                    messages = update.get("messages") if isinstance(update, dict) else None
                    if not messages:
                        continue
                    last = messages[-1]
                    content = getattr(last, "content", None)
                    if content:
                        print(f"\n[{node_name}] {content}")
        except Exception as exc:
            # A failed turn (e.g. quota exhausted even after retries) shouldn't
            # kill the whole CLI - report it and let the user try again.
            print(f"\n[turn failed]\n{_format_error(exc)}")

        print()


if __name__ == "__main__":
    main()
