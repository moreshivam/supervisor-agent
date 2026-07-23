from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command

from supervisor_agent.schemas import GeneratedContent
from supervisor_agent.state import AgentState

# Called once per request (after all research is done), so favor quality over
# speed/rate-limit headroom here - unlike the researcher, which is called
# multiple times per request.
COPYWRITER_MODEL = "llama-3.3-70b-versatile"

PROMPT_PATH = Path(__file__).parent / "prompts" / "copywriter.md"
EXAMPLES_DIR = Path(__file__).parent / "example_content"


@tool
def review_research_reports(state: Annotated[AgentState, InjectedState]) -> str:
    """Read all research reports gathered so far. Call this before writing content."""
    # InjectedState hands this tool the CURRENT graph state directly - the LLM
    # never sees or passes this argument itself. No network/LLM call here,
    # this is just a plain read of what the researcher already saved.
    reports = state.get("research_reports", [])
    if not reports:
        return "No research reports are available yet."
    return "\n\n---\n\n".join(f"## {r['topic']}\n\n{r['report']}" for r in reports)


@tool
def generate_blog_post(
    title: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Save the final blog post."""
    # Same accumulate-into-state pattern as generate_research_report in
    # researcher.py: validate via the shared schema, then write through a
    # Command so it lands in generated_content (operator.add reducer).
    validated = GeneratedContent(content_type="blog", title=title, content=content)
    return Command(
        update={
            "generated_content": [validated.model_dump()],
            "messages": [
                ToolMessage(content=f"Blog post '{title}' saved.", tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def generate_linkedin_post(
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Save the final LinkedIn post."""
    validated = GeneratedContent(content_type="linkedin", content=content)
    return Command(
        update={
            "generated_content": [validated.model_dump()],
            "messages": [
                ToolMessage(content="LinkedIn post saved.", tool_call_id=tool_call_id)
            ],
        }
    )


def _load_prompt() -> str:
    """Read copywriter.md and fill in its placeholders with the example
    content and current datetime."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        linkedin_example=(EXAMPLES_DIR / "linkedin.md").read_text(encoding="utf-8"),
        blog_example=(EXAMPLES_DIR / "blog.md").read_text(encoding="utf-8"),
        current_datetime=datetime.now(timezone.utc).isoformat(),
    )


def build_copywriter_agent():
    """Build the copywriter as a standalone ReAct agent.

    Shares AgentState with the researcher and supervisor, so it can read
    research_reports accumulated earlier and write generated_content that
    flows back up once this node finishes.
    """
    # See researcher.py for why max_retries is set (and why with_retry() isn't used).
    model = ChatGroq(model=COPYWRITER_MODEL, temperature=0.3, max_retries=3)
    return create_react_agent(
        model=model,
        tools=[review_research_reports, generate_blog_post, generate_linkedin_post],
        prompt=_load_prompt(),
        state_schema=AgentState,
        name="copywriter",
    )
