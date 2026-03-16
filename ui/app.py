import uuid
import httpx
import chainlit as cl

# This points to the FastAPI server we configured in launch.json
API_URL = "http://localhost:8000/api/v1/chat"

@cl.on_chat_start
async def on_chat_start():
    """
    Triggered when a new UI session starts.
    Initializes session variables and sends a welcome message.
    """
    # Create a unique thread ID for the LangGraph MongoDB checkpointer
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    
    # Define the role the human is playing. 
    # Let's default to the human acting as the Sales Executive pitching to the AI Customer.
    cl.user_session.set("user_role", "sales_exec")
    
    welcome_message = (
        "Welcome to the Multi-Agent Sales Simulator! 🚀\n\n"
        "You are currently playing the role of the **Sales Executive**, "
        "and the AI will act as the **Customer**.\n\n"
        "To begin, type your opening pitch or greeting!"
    )
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def on_message(message: cl.Message):
    """
    Triggered when the user sends a message in the UI.
    Packages the message and sends it to the FastAPI backend.
    """
    session_id = cl.user_session.get("session_id")
    user_role = cl.user_session.get("user_role")
    
    # Prepare the payload matching the FastAPI ChatRequest Pydantic model
    payload = {
        "session_id": session_id,
        "user_message": message.content,
        "actor": user_role,
        "session_metadata": {
            # You can dynamically alter these to test different scenarios
            "scenario": "Initial discovery call for B2B enterprise software",
            "buyer_persona": "A highly technical, budget-conscious CTO",
            "buyer_pain_points": "High infrastructure costs and vendor lock-in"
        }
    }
    
    # Create an empty message in the UI to show a loading state
    msg = cl.Message(content="")
    await msg.send()
    
    # Call the FastAPI backend asynchronously
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # The backend returns the full conversation history.
            # We extract the very last message (the AI's response) to display in the UI.
            if data.get("messages"):
                latest_response = data["messages"][-1]["content"]
                msg.content = latest_response
                await msg.update()
            else:
                msg.content = "Received an empty response from the simulation."
                await msg.update()
                
        except httpx.HTTPError as e:
            msg.content = f"⚠️ Error communicating with the backend API: {str(e)}"
            await msg.update()