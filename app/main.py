from contextlib import asynccontextmanager
import certifi
from fastapi import FastAPI
from pymongo import MongoClient

from app.core.config import settings
from app.api.routes import router
from app.rag.vectorstore import load_vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes database connections and loads AI models into memory.
    """
    print("Starting up Multi-Agent Sales Simulator...")
    
    # 1. Initialize MongoDB Client
    # We attach it to app.state so it can be accessed globally by endpoints
    app.state.db_client = MongoClient(
        settings.mongodb_uri, 
        tlsCAFile=certifi.where()
    )
    print(f"Connected to MongoDB at {settings.mongodb_uri}")
    
    # 2. Load FAISS Vector Store (if it exists)
    try:
        app.state.faiss_index = load_vector_store()
        print("FAISS vector store loaded successfully.")
    except FileNotFoundError:
        print("Warning: FAISS index not found. RAG context will be empty. Run build_and_save_index() first.")
        app.state.faiss_index = None

    yield # The application runs while yielded

    # 3. Teardown
    print("Shutting down...")
    app.state.db_client.close()
    print("MongoDB connection closed.")

# Initialize the FastAPI app
app = FastAPI(
    title="Sales Simulator API",
    description="Multi-agent agentic AI system for sales training",
    version="0.1.0",
    lifespan=lifespan
)

# Include the routing logic
app.include_router(router)

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "environment": settings.environment}