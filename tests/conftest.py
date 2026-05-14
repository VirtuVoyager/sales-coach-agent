"""
Shared fixtures used across unit, integration, and eval test suites.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.state import SimulationState


# ---------------------------------------------------------------------------
# Message fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def human_message():
    return HumanMessage(content="Hi, I'm interested in your CRM solution.")


@pytest.fixture
def ai_message():
    return AIMessage(content="Great to hear! Our CRM integrates with 500+ tools and has a 30-day free trial.")


@pytest.fixture
def conversation_history(human_message, ai_message):
    return [human_message, ai_message]


# ---------------------------------------------------------------------------
# State fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state(conversation_history) -> SimulationState:
    return {
        "messages": conversation_history,
        "next_agent": "orchestrator",
        "retrieved_context": [
            "Our CRM supports 500+ integrations including Salesforce and HubSpot.",
            "Pricing starts at $49/seat/month with annual billing.",
        ],
        "memory": {
            "customer_id": "cust_001",
            "past_objections": ["too expensive", "integration complexity"],
        },
        "session_metadata": {
            "scenario": "Enterprise CRM pitch",
            "buyer_persona": "VP of Sales at a mid-market SaaS company",
            "buyer_pain_points": "manual pipeline reporting and CRM adoption by reps",
        },
    }


@pytest.fixture
def empty_state() -> SimulationState:
    return {
        "messages": [],
        "next_agent": "orchestrator",
        "retrieved_context": [],
        "memory": {},
        "session_metadata": {},
    }


# ---------------------------------------------------------------------------
# LLM mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Async-capable mock LLM that returns a generic AIMessage by default."""
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(
        content="This is a mock LLM response for testing purposes."
    )
    return llm


@pytest.fixture
def mock_structured_llm():
    """Mock LLM with structured output already bound (for orchestrator)."""
    llm = AsyncMock()
    return llm


# ---------------------------------------------------------------------------
# MongoDB mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mongo_collection():
    collection = MagicMock()
    collection.find_one.return_value = {
        "customer_id": "cust_001",
        "past_objections": ["price", "integration"],
        "industry": "SaaS",
    }
    collection.update_one.return_value = MagicMock(upserted_id=None, modified_count=1)
    return collection


@pytest.fixture
def mock_mongo_db(mock_mongo_collection):
    db = MagicMock()
    db.__getitem__.return_value = mock_mongo_collection
    return db


@pytest.fixture
def mock_mongo_client(mock_mongo_db):
    client = MagicMock()
    client.__getitem__.return_value = mock_mongo_db
    return client


# ---------------------------------------------------------------------------
# FAISS mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_faiss_index():
    faiss = MagicMock()
    faiss.similarity_search.return_value = [
        MagicMock(page_content="Product supports SSO and SAML 2.0."),
        MagicMock(page_content="Free onboarding for enterprise plans."),
    ]
    return faiss
