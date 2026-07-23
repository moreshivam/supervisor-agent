import operator
from typing import Annotated, NotRequired

from langgraph.graph import MessagesState
from langgraph.managed.is_last_step import RemainingSteps


# MessagesState already provides `messages: Annotated[list[BaseMessage], add_messages]`.
# All three agents (supervisor, researcher, copywriter) are built with this
# SAME state_schema, so when one is compiled as a node inside another's graph,
# LangGraph merges these shared keys back up automatically once that node
# finishes running.
class AgentState(MessagesState):
    """Shared state across the supervisor, researcher, and copywriter nodes.

    research_reports accumulates across multiple researcher calls (one report
    per call) instead of the last call overwriting the rest.

    remaining_steps is required by create_react_agent's internal step-limit check.
    """

    research_reports: Annotated[list[dict], operator.add]
    # Same accumulate-don't-overwrite pattern as research_reports, in case the
    # copywriter is ever called more than once (e.g. blog + linkedin) in a request.
    generated_content: Annotated[list[dict], operator.add]
    remaining_steps: NotRequired[RemainingSteps]
