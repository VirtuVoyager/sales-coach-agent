"""
Integration tests for POST /api/v1/feedback.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


STARTUP_PATCHES = [
    patch("app.main.MongoClient"),
    patch("app.main.load_vector_store", return_value=None),
    patch("certifi.where", return_value="/mock/ca-bundle.crt"),
]


@pytest.fixture
def client():
    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in STARTUP_PATCHES:
            stack.enter_context(p)
        with patch("app.api.routes.Opik") as MockOpik:
            mock_opik_instance = MagicMock()
            MockOpik.return_value = mock_opik_instance

            from app.main import app
            with TestClient(app) as c:
                yield c, mock_opik_instance


@pytest.fixture
def positive_feedback():
    return {
        "session_id": "test-session-fb-001",
        "value": 1.0,
        "comment": "Great pitch, handled objections well.",
    }


@pytest.fixture
def negative_feedback():
    return {
        "session_id": "test-session-fb-002",
        "value": 0.0,
        "comment": "Didn't address budget concerns.",
    }


class TestFeedbackEndpointSuccess:
    def test_positive_feedback_returns_200(self, client, positive_feedback):
        c, _ = client
        resp = c.post("/api/v1/feedback", json=positive_feedback)
        assert resp.status_code == 200

    def test_negative_feedback_returns_200(self, client, negative_feedback):
        c, _ = client
        resp = c.post("/api/v1/feedback", json=negative_feedback)
        assert resp.status_code == 200

    def test_response_confirms_logged(self, client, positive_feedback):
        c, _ = client
        resp = c.post("/api/v1/feedback", json=positive_feedback)
        body = resp.json()
        assert "logged" in body.get("status", "").lower()

    def test_opik_trace_called(self, client, positive_feedback):
        c, mock_opik_instance = client
        c.post("/api/v1/feedback", json=positive_feedback)
        mock_opik_instance.trace.assert_called_once()

    def test_opik_trace_includes_session_id(self, client, positive_feedback):
        c, mock_opik_instance = client
        c.post("/api/v1/feedback", json=positive_feedback)
        call_kwargs = mock_opik_instance.trace.call_args.kwargs
        assert call_kwargs["metadata"]["session_id"] == positive_feedback["session_id"]

    def test_opik_trace_includes_score(self, client, positive_feedback):
        c, mock_opik_instance = client
        c.post("/api/v1/feedback", json=positive_feedback)
        call_kwargs = mock_opik_instance.trace.call_args.kwargs
        assert call_kwargs["output"]["score"] == 1.0

    def test_opik_trace_includes_comment(self, client, positive_feedback):
        c, mock_opik_instance = client
        c.post("/api/v1/feedback", json=positive_feedback)
        call_kwargs = mock_opik_instance.trace.call_args.kwargs
        assert "Great pitch" in call_kwargs["output"]["comment"]

    def test_feedback_without_comment_accepted(self, client):
        c, _ = client
        payload = {"session_id": "test-session-fb-003", "value": 1.0}
        resp = c.post("/api/v1/feedback", json=payload)
        assert resp.status_code == 200


class TestFeedbackEndpointValidation:
    def test_missing_session_id_returns_422(self, client):
        c, _ = client
        resp = c.post("/api/v1/feedback", json={"value": 1.0})
        assert resp.status_code == 422

    def test_missing_value_returns_422(self, client):
        c, _ = client
        resp = c.post("/api/v1/feedback", json={"session_id": "s1"})
        assert resp.status_code == 422


class TestFeedbackEndpointOpikError:
    def test_opik_exception_returns_500(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in STARTUP_PATCHES:
                stack.enter_context(p)
            with patch("app.api.routes.Opik") as MockOpik:
                mock_opik_instance = MagicMock()
                mock_opik_instance.trace.side_effect = ConnectionError("Opik unreachable")
                MockOpik.return_value = mock_opik_instance

                from app.main import app
                with TestClient(app) as c:
                    resp = c.post(
                        "/api/v1/feedback",
                        json={"session_id": "s1", "value": 1.0},
                    )
                    assert resp.status_code == 500
