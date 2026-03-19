from typing import Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from opik import track

@track(name="Generate Customer Prompt")
def get_customer_prompt(session_metadata: dict[str, Any] | None = None) -> ChatPromptTemplate:
    """
    Dynamically generates the Customer Agent prompt.
    Allows for injecting specific buyer personas or constraints at runtime.
    """
    # Default persona and concerns if none are provided in the state
    persona = "a pragmatic B2B software buyer"
    pain_points = "budget constraints and integration complexities"
    
    if session_metadata:
        persona = session_metadata.get("buyer_persona", persona)
        pain_points = session_metadata.get("buyer_pain_points", pain_points)
    
    system_instructions = f"""You are acting as {persona} in a simulated sales conversation.
Your primary pain points and concerns are: {pain_points}.

Your role is to evaluate the product being pitched by the sales executive. 
- Ask probing questions about features, pricing, and ROI.
- Push back on claims that seem too good to be true.
- Raise objections naturally based on your pain points.
- Do not agree to a purchase immediately; require persuasion and clear value.

Respond naturally, professionally, and concisely to the latest message. Act exactly like a real prospect on a sales call.
"""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instructions),
        MessagesPlaceholder(variable_name="messages")
    ])