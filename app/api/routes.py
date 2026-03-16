from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.state import SimulationState
from app.agents.orchestrator.graph import SimulationOrchestrator
from app.database.checkpointer import get_checkpointer

router = APIRouter(prefix="/api/v1", tags=["Simulation"])

class ChatRequest(BaseModel):
    """Payload expected from the frontend UI."""
    session_id: str = Field(..., description="Unique ID for the conversation thread")
    user_message: str | None = Field(None, description="The message from the human user")
    actor: str = Field("customer", description="The role the human is playing: 'customer' or 'sales_exec'")
    session_metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata like buyer_persona or product details")

class ChatResponse(BaseModel):
    """Payload returned to the frontend UI."""
    session_id: str
    messages: list[dict[str, str]]
    next_agent: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, api_request: Request):
    """
    Processes a chat turn through the multi-agent LangGraph system.
    """
    try:
        # Retrieve global resources initialized in main.py
        db_client = api_request.app.state.db_client
        faiss_index = api_request.app.state.faiss_index
        
        # 1. Setup Checkpointer and Orchestrator
        checkpointer = get_checkpointer(db_client)
        orchestrator = SimulationOrchestrator(checkpointer=checkpointer)
        
        # 2. Configure the thread for LangGraph memory
        config = {"configurable": {"thread_id": request.session_id}}
        
        # 3. Prepare the state updates
        # If the user sent a message, wrap it in a HumanMessage. 
        # If the user is just kicking off the simulation, messages can be empty.
        input_messages = [HumanMessage(content=request.user_message)] if request.user_message else []
        
        # We perform a dummy retrieval here if FAISS is loaded. 
        # In a real scenario, you'd use the user_message to search FAISS.
        retrieved_docs = []
        if faiss_index and request.user_message:
            docs = faiss_index.similarity_search(request.user_message, k=3)
            retrieved_docs = [doc.page_content for doc in docs]

        # 4. Define the input state for this turn
        input_state: SimulationState = {
            "messages": input_messages,
            "next_agent": "orchestrator", # Always start with the orchestrator
            "retrieved_context": retrieved_docs,
            "memory": {}, # You would hook up MongoManager here to pull long-term memory
            "session_metadata": request.session_metadata
        }
        
        # 5. Execute the Graph
        # We use ainvoke to run the compiled graph. It will route until it hits a break or finishes its logic.
        final_state = await orchestrator.ainvoke(input_state, config)
        
        # 6. Format the response for the UI
        # Extract the latest messages added to the state
        formatted_messages = []
        for msg in final_state.get("messages", []):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            formatted_messages.append({"role": role, "content": msg.content})
            
        return ChatResponse(
            session_id=request.session_id,
            messages=formatted_messages,
            next_agent=final_state.get("next_agent", "unknown")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))