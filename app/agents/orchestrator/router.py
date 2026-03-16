from enum import StrEnum
from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END

from app.agents.base_node import BaseNode
from app.agents.state import SimulationState
from app.prompts.agents.orchestrator import get_orchestrator_prompt
from langchain_core.runnables.config import RunnableConfig

class AgentRoute(StrEnum):
    """Enumeration of valid routing destinations."""
    SALES_EXEC = "sales_exec"
    CUSTOMER = "customer"
    END_ROUTE = "end"

class RouteDecision(BaseModel):
    """Schema to force the LLM to output a precise routing decision."""
    next_agent: AgentRoute = Field(
        description="The exact name of the next agent to act. Must be 'sales_exec', 'customer', or 'end'."
    )

class OrchestratorNode(BaseNode):
    """
    Analyzes the conversation state and routes execution to the correct sub-agent.
    """
    def __init__(self, name: str, llm: BaseChatModel):
        super().__init__(name)
        # Bind the structured output schema to the Azure OpenAI client
        self.llm = llm.with_structured_output(RouteDecision)
        
    async def ainvoke(self, state: SimulationState, config: RunnableConfig | None = None) -> Command:
        # 1. Fetch and format the dynamic prompt with runtime arguments
        prompt_template = get_orchestrator_prompt(state.get("session_metadata"))
        formatted_prompt = await prompt_template.ainvoke({"messages": state["messages"]})
        
        # Invoke the LLM to get the structured routing decision
        decision: RouteDecision = await self.llm.ainvoke(formatted_prompt)
        
        # Map the Enum value to LangGraph's END constant or extract the string value
        goto_target = END if decision.next_agent == AgentRoute.END_ROUTE else decision.next_agent.value
        
        return Command(
            goto=goto_target,
            # We store the string value in the state so it remains easily JSON serializable
            update={"next_agent": decision.next_agent.value} 
        )