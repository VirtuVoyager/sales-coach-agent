from typing import Any
import certifi
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

class MongoManager:
    """
    Manages connections and operations for MongoDB long-term memory storage.
    """
    def __init__(self, uri: str, db_name: str):
        # Initialize the client. In a FastAPI app, this should ideally be 
        # instantiated once on startup and reused across requests.
        self.client: MongoClient[dict[str, Any]] = MongoClient(
            uri, 
            tlsCAFile=certifi.where()
        )
        self.db: Database[dict[str, Any]] = self.client[db_name]
        
    def get_collection(self, collection_name: str) -> Collection[dict[str, Any]]:
        """Utility to fetch a specific collection."""
        return self.db[collection_name]
        
    def get_customer_memory(self, customer_id: str) -> dict[str, Any] | None:
        """
        Retrieves historical insights and long-term memory for a specific customer.
        Returns a dictionary of traits, past objections, or None if new.
        """
        collection = self.get_collection("customer_memory")
        # Exclude the MongoDB '_id' ObjectId to keep the dictionary JSON serializable
        return collection.find_one({"customer_id": customer_id}, {"_id": 0})
        
    def update_customer_memory(self, customer_id: str, new_data: dict[str, Any]) -> None:
        """
        Upserts new memory points into the customer's long-term storage.
        """
        collection = self.get_collection("customer_memory")
        collection.update_one(
            {"customer_id": customer_id},
            {"$set": new_data},
            upsert=True # Creates the document if the customer doesn't exist yet
        )

    def close(self) -> None:
        """Closes the connection pool."""
        self.client.close()