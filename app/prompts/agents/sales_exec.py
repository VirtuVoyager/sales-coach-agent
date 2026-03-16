from typing import Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder

def get_sales_exec_prompt(
    retrieved_context: list[str], 
    memory: dict[str, Any] | None = None
) -> ChatPromptTemplate:
    """
    Dynamically generates the Sales Executive prompt, injecting RAG context and MongoDB memory.
    """
    # Format the RAG context
    context_str = "\n".join(retrieved_context) if retrieved_context else "No specific product context available."
    
    # Format long-term memory/history
    memory_str = str(memory) if memory else "No prior interaction history with this customer."
    
    system_instructions = f"""You are an expert, highly strategic B2B Sales Executive.
Your goal is to understand the customer's needs, pitch your product effectively, handle objections, and move the deal forward.

Use the following Product Knowledge (retrieved from your internal database) to ground your answers:
<product_knowledge>
{context_str}
</product_knowledge>

Use the following Customer History to personalize your approach:
<customer_history>
{memory_str}
</customer_history>

Respond professionally, persuasively, and concisely to the latest message. Do not break character.
"""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instructions),
        MessagesPlaceholder(variable_name="messages")
    ])