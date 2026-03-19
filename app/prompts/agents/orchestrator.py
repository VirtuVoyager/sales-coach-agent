from typing import Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from opik import track

@track(name="Generate Orchestrator Prompt")
def get_orchestrator_prompt(session_metadata: dict[str, Any] | None = None) -> ChatPromptTemplate:
    scenario = session_metadata.get("scenario", "a general B2B sales call") if session_metadata else "a general B2B sales call"
    
    system_instructions = f"""You are the Orchestrator for a multi-agent sales simulation.
The current simulation scenario is: {scenario}.

Your ONLY job is to analyze the conversation history and determine who should speak next.
- If the human user is acting as the customer and just spoke, route to 'sales_exec'.
- If the human user is acting as the sales executive and just spoke, route to 'customer'.
- IMPORTANT: If the AI just finished speaking and it is now the human user's turn to reply, you MUST choose 'end'. This pauses the simulation so the human can type.
- If running in fully automated agent-to-agent mode, alternate between 'sales_exec' and 'customer', but use 'end' if a natural conclusion is reached.

Analyze the flow and choose the appropriate next agent."""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instructions),
        MessagesPlaceholder(variable_name="messages")
    ])