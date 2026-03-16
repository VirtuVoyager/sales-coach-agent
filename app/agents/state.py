from typing import Annotated, TypedDict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SimulationState(TypedDict):
    """
    Represents the shared state of the multi-agent sales simulator.
    Passed between the Orchestrator, Sales Exec, Customer, and Tool nodes.
    """
    
    # The complete conversation history. 
    # 'add_messages' ensures new messages are appended to the list, not overwritten.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Tracks which entity should act next. Used by the Orchestrator to route edges.
    # Expected values: 'sales_exec', 'customer', 'tool_node', or 'end'
    next_agent: str
    
    # Stores documents retrieved from the FAISS (gte-base) vector store
    retrieved_context: list[str]
    
    # Stores long-term memory or session data retrieved from MongoDB
    memory: dict[str, Any]
    
    # Optional metadata tracking the current simulation parameters (e.g., product being sold)
    session_metadata: dict[str, Any]