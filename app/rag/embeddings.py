from langchain_huggingface import HuggingFaceEmbeddings
import torch

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes the Hugging Face embedding model (gte-base).
    Automatically detects Apple Silicon (mps) for hardware acceleration.
    """
    # Detect if Apple Silicon (M-series) is available, otherwise fallback to CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    return HuggingFaceEmbeddings(
        model_name="thenlper/gte-base",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True} # Normalization improves cosine similarity accuracy
    )