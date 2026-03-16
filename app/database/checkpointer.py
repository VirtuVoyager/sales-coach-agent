from typing import Any
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from app.core.config import settings

def get_checkpointer(client: MongoClient[dict[str, Any]]) -> MongoDBSaver:
    """
    Initializes the official MongoDB Checkpointer for LangGraph.
    This provides short-term (turn-by-turn) memory and state persistence across UI interactions.
    
    Args:
        client: The active PyMongo client connection.
        
    Returns:
        MongoDBSaver: The checkpointer instance to be passed into the LangGraph compiler.
    """
    # The MongoDBSaver automatically creates the required collections (e.g., "checkpoints")
    # within your specified database to track thread states.
    return MongoDBSaver(client, db_name=settings.mongodb_db_name)