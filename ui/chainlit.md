# Welcome to the Sales Simulator 🎯

This interface connects directly to your custom LangGraph multi-agent backend.

### How it works:
- **You** act as the Sales Executive.
- **The AI** acts as the Customer, using a dynamic persona injected via the backend metadata.
- The **Orchestrator Node** analyzes every turn and dynamically routes the execution.
- Context is retrieved locally using **Hugging Face gte-base** embeddings and **FAISS**.
- Short-term conversation memory is persisted automatically using the **MongoDB Checkpointer**.

*Tip: You can modify the `session_metadata` in `ui/app.py` to test how the agent reacts to different buyer personas!*