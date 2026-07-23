import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION

from supervisor_agent.copywriter import build_copywriter_agent
from supervisor_agent.researcher import build_researcher_agent
from supervisor_agent.state import AgentState

# Called once per request, so favor quality (matches researcher.py/copywriter.py's reasoning).
SUPERVISOR_MODEL = "llama-3.3-70b-versatile"

PROMPT_PATH = Path(__file__).parent / "prompts" / "supervisor.md"


def _remove_non_handoff_tool_calls(last_ai_message: AIMessage, handoff_tool_call_id: str) -> AIMessage:
    """Keep only this handoff's tool call on the triggering AIMessage.

    If the supervisor's turn included multiple tool calls, every one of them
    needs exactly one matching ToolMessage or the message history becomes
    invalid for the next LLM call - so each parallel handoff gets its own
    trimmed copy of that AIMessage.
    """
    return AIMessage(
        content=last_ai_message.content,
        tool_calls=[tc for tc in last_ai_message.tool_calls if tc["id"] == handoff_tool_call_id],
        name=last_ai_message.name,
        id=str(uuid.uuid4()),
    )


def _make_handoff_tool(agent_name: str) -> BaseTool:
    """Build a handoff tool that carries an explicit, atomic task_description.

    langgraph-supervisor's default transfer_to_<agent> tool just forwards the
    whole conversation and lets the sub-agent infer its task. supervisor.md's
    plan relies on the supervisor issuing a distinct, explicit task per call
    (e.g. one of 2-4 atomic research angles), so a custom handoff tool per
    agent is used instead - this is an officially supported extension point:
    passing tools whose metadata marks them as handoff destinations skips
    langgraph-supervisor's auto-generated ones for that agent.
    """

    @tool(
        f"handoff_to_{agent_name}",
        description=f"Assign a specific, atomic task to the {agent_name} agent.",
    )
    def handoff(
        task_description: str,
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = ToolMessage(
            content=f"Handed off to {agent_name}.",
            tool_call_id=tool_call_id,
        )
        task_message = HumanMessage(content=task_description)
        last_ai_message = state["messages"][-1]

        # Mirrors langgraph_supervisor.handoff.create_handoff_tool: if the
        # supervisor's LLM emitted more than one tool call in this turn (e.g.
        # handing off to two agents "simultaneously"), plain Command(goto=...)
        # from each tool call would race - only one goto would actually take
        # effect. Send lets LangGraph's ToolNode fan out to every destination
        # safely within the same superstep instead.
        if len(last_ai_message.tool_calls) > 1:
            handoff_messages = [
                *state["messages"][:-1],
                _remove_non_handoff_tool_calls(last_ai_message, tool_call_id),
                tool_message,
                task_message,
            ]
            return Command(
                graph=Command.PARENT,
                goto=[Send(agent_name, {**state, "messages": handoff_messages})],
            )

        # Single handoff: no race to guard against, update the parent graph directly.
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": [tool_message, task_message]},
        )

    handoff.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff


def _load_prompt() -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(current_datetime=datetime.now(timezone.utc).isoformat())


def build_supervisor_agent():
    """Build the full multi-agent graph: supervisor + researcher + copywriter.

    All three share AgentState, so research_reports/generated_content
    accumulated by the sub-agents flow back up into the supervisor's state
    once each node finishes. Compiled with an in-memory checkpointer so a
    conversation (thread_id) can span multiple turns without losing state.
    """
    researcher = build_researcher_agent()
    copywriter = build_copywriter_agent()

    workflow = create_supervisor(
        agents=[researcher, copywriter],
        # See researcher.py for why max_retries is set (and why with_retry() isn't used).
        model=ChatGroq(model=SUPERVISOR_MODEL, temperature=0, max_retries=3),
        tools=[_make_handoff_tool("researcher"), _make_handoff_tool("copywriter")],
        prompt=_load_prompt(),
        state_schema=AgentState,
    )
    return workflow.compile(checkpointer=InMemorySaver())
