from typing import Any
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig

from app.agents.base_node import BaseNode
from app.agents.state import SimulationState
from app.prompts.agents.sales_exec import get_sales_exec_prompt

class SalesExecNode(BaseNode):
    """
    Executes the Sales Executive logic, leveraging RAG context and long-term memory.
    """
    def __init__(self, name: str, llm: BaseChatModel):
        super().__init__(name)
        self.llm = llm
        
    async def ainvoke(self, state: SimulationState, config: RunnableConfig | None = None) -> Command:
        # 1. Fetch dynamic prompt injected with context and memory from the state
        prompt_template = get_sales_exec_prompt(
            retrieved_context=state.get("retrieved_context", []),
            memory=state.get("memory", {})
        )
        
        # 2. Format the prompt with the ongoing conversation history
        formatted_prompt = await prompt_template.ainvoke({"messages": state["messages"]})
        
        # 3. Call the Azure OpenAI LLM
        response: AIMessage = await self.llm.ainvoke(formatted_prompt)
        
        # 4. Return a LangGraph Command. 
        # - The 'update' key appends the new AIMessage to the state's messages list.
        # - The 'goto' key routes the flow back to the orchestrator to decide the next move.
        return Command(
            update={"messages": [response]},
            goto="orchestrator"
        )