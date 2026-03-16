# Multi-Agent Sales Simulator 🎯

A highly modular, agentic AI system designed to simulate realistic B2B sales conversations. Built with modern Python (>=3.13), this application leverages LangGraph to orchestrate dynamic interactions between a simulated Sales Executive and a Customer, utilizing Retrieval-Augmented Generation (RAG) and long-term memory.

---

## 🚀 Key Features

* **Multi-Agent Orchestration:** A central Orchestrator dynamically routes conversations between the human user, the AI Sales Executive, and the AI Customer using strict `StrEnum` routing.
* **Flexible LLM Support:** Seamlessly toggle between **Azure OpenAI** and **Groq** via environment variables without changing a single line of code.
* **Advanced RAG Pipeline:** Ingest `.pdf` and `.txt` product sheets into a local **FAISS** vector store using **Hugging Face (`gte-base`)** embeddings (with automatic Apple Silicon `mps` acceleration).
* **Dual Memory Architecture:** * **Short-Term:** Official LangGraph MongoDB Checkpointer maintains precise turn-by-turn state across UI sessions.
  * **Long-Term:** MongoDB persistence for storing and retrieving historical customer profiles and prior objections.
* **Decoupled Stack:** A robust **FastAPI** backend powering a highly responsive **Chainlit** frontend.
* **Lightning Fast Setup:** Managed entirely by `uv` for instantaneous dependency resolution and virtual environment creation.

---

## 🛠️ Prerequisites

* **Python:** `>= 3.13`
* **Package Manager:** `uv`
* **Database:** MongoDB (Local or Atlas)

---

## ⚙️ Installation & Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/VirtuVoyager/sales-coach-agent.git
   ```

2. **Install dependencies and create the virtual environment:**
   ```bash
   uv sync
   ```

3. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Configure Environment Variables:**
   Copy the template and fill in your API keys and MongoDB URI.
   ```bash
   cp .env.template .env
   ```
   *Note: Set `LLM_PROVIDER` to either `azure` or `groq` depending on your preference.*

---

## 📚 Data Ingestion (RAG)

Before running the simulation, you need to populate the FAISS index with product knowledge.

1. Place your product `.pdf` or `.txt` files into `data/sample_docs/`.
2. Open a Python interactive shell and run the ingestion script:
   ```python
   from app.rag.vectorstore import build_and_save_index
   build_and_save_index()
   ```
   *This will chunk the documents, generate embeddings, and save the binary index to `data/faiss_index/`.*

---

## 🏃‍♂️ Running the Application

### Option 1: VS Code (Recommended)
This project includes a `.vscode/launch.json` file for integrated debugging.
1. Open the **Run and Debug** panel in VS Code.
2. Select **"Run Full Stack"** from the dropdown menu.
3. Hit Play. This launches the FastAPI backend and Chainlit UI simultaneously.

### Option 2: Terminal
You will need two separate terminal windows (with the `.venv` activated in both).

**Terminal 1: Start the FastAPI Backend**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: Start the Chainlit UI**
```bash
chainlit run ui/app.py -w --port 8001
```

Once running, navigate to `http://localhost:8001` in your browser to start the simulation!

---

## 📁 Project Structure

```text
sales_simulator/
├── .vscode/               # Debugger configurations for Full Stack execution
├── app/                   # FastAPI Backend & LangGraph Engine
│   ├── api/               # REST endpoints
│   ├── core/              # Config (Pydantic) and LLM Factory
│   ├── database/          # MongoDB Checkpointer and Long-Term Memory
│   ├── rag/               # Document ingestion, HF Embeddings, and FAISS
│   ├── agents/            # LangGraph Nodes and Base Classes
│   │   ├── orchestrator/  # Dynamic router (StrEnum) and Graph Compiler
│   │   ├── sales_exec/    # RAG-enabled Sales Agent
│   │   ├── customer/      # Persona-driven Customer Agent
│   ├── prompts/           # Dynamic .py prompt generators
├── data/                  # Local storage for sample docs and FAISS indexes
├── ui/                    # Chainlit Frontend application
├── pyproject.toml         # uv dependency definitions
└── .env                   # Environment variables (Azure, Groq, Mongo)
```