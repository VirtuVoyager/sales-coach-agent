from typing import Any
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig

from app.agents.base_node import BaseNode
from app.agents.state import SimulationState
from app.prompts.agents.customer import get_customer_prompt

class CustomerNode(BaseNode):
    """
    Executes the Customer Agent logic, acting as the buyer in the simulation.
    """
    def __init__(self, name: str, llm: BaseChatModel):
        super().__init__(name)
        self.llm = llm
        
    async def ainvoke(self, state: SimulationState, config: RunnableConfig | None = None) -> Command:
        # 1. Fetch the dynamic prompt, injecting any specific persona details from metadata
        prompt_template = get_customer_prompt(
            session_metadata=state.get("session_metadata")
        )
        
        # 2. Format the prompt with the ongoing conversation history
        formatted_prompt = await prompt_template.ainvoke({"messages": state["messages"]})
        
        # 3. Call the Azure OpenAI LLM
        response: AIMessage = await self.llm.ainvoke(formatted_prompt)
        
        # 4. Return a LangGraph Command.
        # - Appends the new AIMessage to the state's messages list.
        # - Routes the flow back to the orchestrator to decide the next move.
        return Command(
            update={"messages": [response]},
            goto="orchestrator"
        )