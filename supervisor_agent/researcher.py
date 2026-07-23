from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilyExtract, TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from supervisor_agent.schemas import ResearchReport
from supervisor_agent.state import AgentState

# Small/fast models are cheaper on Groq's free-tier rate limits, but they were
# unreliable at formatting a large multi-paragraph tool-call argument (tested:
# llama-3.1-8b-instant threw tool_use_failed). This model is called once per
# research angle (2-4x per user request), so the larger model is worth it.
RESEARCHER_MODEL = "llama-3.3-70b-versatile"

PROMPT_PATH = Path(__file__).parent / "prompts" / "researcher.md"

# Tavily tools: one for web search, one for pulling full page content from a
# specific URL. Both map directly to the tool names referenced in researcher.md.
search_web = TavilySearch(max_results=5, name="search_web")
extract_content_from_webpage = TavilyExtract(name="extract_content_from_webpage")


@tool
def generate_research_report(
    topic: str,
    report: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Save the final research report for a topic so the copywriter can use it later."""
    # Validate/normalize through the shared Pydantic schema before it enters
    # graph state, so the copywriter can trust its shape later.
    validated = ResearchReport(topic=topic, report=report)

    # Returning a Command (instead of a plain string) lets this tool write
    # directly into the graph's shared state. `research_reports` uses an
    # operator.add reducer (see state.py), so each call APPENDS rather than
    # overwriting previous research from earlier calls in this same request.
    return Command(
        update={
            "research_reports": [validated.model_dump()],
            # The ReAct loop still needs a ToolMessage reply to the tool call
            # that triggered this, or the agent's message history breaks.
            "messages": [
                ToolMessage(
                    content=f"Research report on '{topic}' saved.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def _load_prompt() -> str:
    """Read researcher.md and fill in the {current_datetime} placeholder."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(current_datetime=datetime.now(timezone.utc).isoformat())


def build_researcher_agent():
    """Build the researcher as a standalone ReAct agent (LLM <-> tools loop).

    This gets compiled once and added as a single node inside the
    supervisor's graph later. Because it shares AgentState with the
    supervisor, research_reports accumulated here flow back up automatically
    when this node finishes.
    """
    # max_retries: Groq/Llama tool-calling occasionally malforms a tool call
    # (e.g. jamming JSON args into the tool name string), which surfaces as a
    # BadRequestError. Retrying the LLM call alone has a good chance of
    # getting a well-formed call on the next attempt. (Runnable.with_retry()
    # doesn't work here - it wraps the model in a RunnableRetry that no
    # longer exposes bind_tools(), which create_react_agent needs.)
    model = ChatGroq(model=RESEARCHER_MODEL, temperature=0, max_retries=3)
    return create_react_agent(
        model=model,
        tools=[search_web, extract_content_from_webpage, generate_research_report],
        prompt=_load_prompt(),
        state_schema=AgentState,
        name="researcher",
    )
