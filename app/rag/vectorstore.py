from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model
from app.rag.ingestion import load_and_chunk_documents

INDEX_PATH = Path("data/faiss_index")
DATA_DIR = Path("data/sample_docs")

def build_and_save_index() -> FAISS | None:
    """
    Loads documents, generates embeddings, builds the FAISS index, and saves it to disk.
    Run this function whenever you add new documents to the sample_docs folder.
    """
    print("Loading and chunking documents...")
    documents = load_and_chunk_documents(DATA_DIR)
    
    if not documents:
        return None
        
    print(f"Generated {len(documents)} chunks. Building FAISS index...")
    embeddings = get_embedding_model()
    
    # Build standard FlatL2 index (exact search, no data loss)
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Ensure the directory exists and save the index
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_PATH))
    print(f"Index successfully saved to {INDEX_PATH}")
    
    return vector_store

def load_vector_store() -> FAISS:
    """
    Loads the persisted FAISS index from disk for use during the simulation.
    """
    embeddings = get_embedding_model()
    
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No FAISS index found at {INDEX_PATH}. Please run build_and_save_index() first.")
        
    # allow_dangerous_deserialization is required in recent LangChain updates when loading local pickle files
    return FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)