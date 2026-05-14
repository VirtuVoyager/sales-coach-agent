"""
Integration tests for POST /api/v1/chat.

Strategy: patch MongoClient and load_vector_store at the app startup level so
the lifespan context manager completes cleanly, then patch SimulationOrchestrator
so no real LLM calls are made.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage


PATCHES = [
    patch("app.main.MongoClient"),
    patch("app.main.load_vector_store", return_value=None),
    patch("certifi.where", return_value="/mock/ca-bundle.crt"),
]


def _apply_patches():
    from contextlib import ExitStack
    stack = ExitStack()
    for p in PATCHES:
        stack.enter_context(p)
    return stack


@pytest.fixture
def client_with_mock_orchestrator():
    """TestClient with lifespan mocked and orchestrator returning a canned state."""
    mock_final_state = {
        "messages": [
            HumanMessage(content="Tell me about pricing."),
            AIMessage(content="Our pricing starts at $49/seat/month."),
        ],
        "next_agent": "end",
    }

    with _apply_patches():
        with patch("app.api.routes.SimulationOrchestrator") as MockOrch, \
             patch("app.api.routes.get_checkpointer") as mock_cp, \
             patch("app.api.routes.OpikTracer"), \
             patch("app.api.routes.retrieve_documents", return_value=[]):
            mock_instance = AsyncMock()
            mock_instance.ainvoke.return_value = mock_final_state
            MockOrch.return_value = mock_instance
            mock_cp.return_value = MagicMock()

            from app.main import app
            with TestClient(app) as client:
                yield client


@pytest.fixture
def chat_payload():
    return {
        "session_id": "test-session-001",
        "user_message": "Tell me about your pricing.",
        "actor": "customer",
        "session_metadata": {
            "scenario": "CRM demo",
            "buyer_persona": "VP of Sales",
        },
    }


class TestChatEndpointSuccess:
    def test_returns_200(self, client_with_mock_orchestrator, chat_payload):
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=chat_payload)
        assert resp.status_code == 200

    def test_response_contains_session_id(self, client_with_mock_orchestrator, chat_payload):
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=chat_payload)
        body = resp.json()
        assert body["session_id"] == chat_payload["session_id"]

    def test_response_contains_messages(self, client_with_mock_orchestrator, chat_payload):
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=chat_payload)
        body = resp.json()
        assert isinstance(body["messages"], list)
        assert len(body["messages"]) > 0

    def test_messages_have_role_and_content(self, client_with_mock_orchestrator, chat_payload):
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=chat_payload)
        for msg in resp.json()["messages"]:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("user", "assistant")

    def test_response_contains_next_agent(self, client_with_mock_orchestrator, chat_payload):
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=chat_payload)
        body = resp.json()
        assert "next_agent" in body

    def test_human_messages_mapped_to_user_role(self, client_with_mock_orchestrator, chat_payload):
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=chat_payload)
        roles = [m["role"] for m in resp.json()["messages"]]
        assert "user" in roles

    def test_ai_messages_mapped_to_assistant_role(self, client_with_mock_orchestrator, chat_payload):
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=chat_payload)
        roles = [m["role"] for m in resp.json()["messages"]]
        assert "assistant" in roles


class TestChatEndpointNoMessage:
    def test_empty_user_message_accepted(self, client_with_mock_orchestrator):
        payload = {
            "session_id": "test-session-002",
            "user_message": None,
            "actor": "customer",
            "session_metadata": {},
        }
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200

    def test_missing_optional_fields_use_defaults(self, client_with_mock_orchestrator):
        payload = {
            "session_id": "test-session-003",
        }
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200


class TestChatEndpointValidation:
    def test_missing_session_id_returns_422(self, client_with_mock_orchestrator):
        payload = {"user_message": "Hello", "actor": "customer"}
        resp = client_with_mock_orchestrator.post("/api/v1/chat", json=payload)
        assert resp.status_code == 422


class TestChatEndpointOrchestratorError:
    def test_orchestrator_exception_returns_500(self):
        with _apply_patches():
            with patch("app.api.routes.SimulationOrchestrator") as MockOrch, \
                 patch("app.api.routes.get_checkpointer") as mock_cp, \
                 patch("app.api.routes.OpikTracer"), \
                 patch("app.api.routes.retrieve_documents", return_value=[]):
                mock_instance = AsyncMock()
                mock_instance.ainvoke.side_effect = RuntimeError("LLM timeout")
                MockOrch.return_value = mock_instance
                mock_cp.return_value = MagicMock()

                from app.main import app
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/chat",
                        json={"session_id": "err-session", "user_message": "Hi"},
                    )
                    assert resp.status_code == 500
