"""Unit tests for app/agents/state.py — SimulationState schema and message merging."""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

from app.agents.state import SimulationState


class TestSimulationStateKeys:
    def test_has_messages_key(self):
        assert "messages" in SimulationState.__annotations__

    def test_has_next_agent_key(self):
        assert "next_agent" in SimulationState.__annotations__

    def test_has_retrieved_context_key(self):
        assert "retrieved_context" in SimulationState.__annotations__

    def test_has_memory_key(self):
        assert "memory" in SimulationState.__annotations__

    def test_has_session_metadata_key(self):
        assert "session_metadata" in SimulationState.__annotations__


class TestAddMessagesReducer:
    """Validate that add_messages appends rather than overwrites."""

    def test_add_messages_appends_new_message(self):
        existing = [HumanMessage(content="Hello")]
        new = [AIMessage(content="Hi there!")]
        result = add_messages(existing, new)
        assert len(result) == 2
        assert result[1].content == "Hi there!"

    def test_add_messages_preserves_order(self):
        msgs = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
        ]
        result = add_messages([], msgs)
        contents = [m.content for m in result]
        assert contents == ["msg1", "msg2", "msg3"]

    def test_add_messages_empty_new_list(self):
        existing = [HumanMessage(content="Hello")]
        result = add_messages(existing, [])
        assert len(result) == 1

    def test_add_messages_both_empty(self):
        result = add_messages([], [])
        assert result == []

    def test_add_messages_multiple_in_sequence(self):
        state_msgs = [HumanMessage(content="Turn 1")]
        state_msgs = add_messages(state_msgs, [AIMessage(content="Turn 2")])
        state_msgs = add_messages(state_msgs, [HumanMessage(content="Turn 3")])
        assert len(state_msgs) == 3
        assert state_msgs[2].content == "Turn 3"


class TestStateConstruction:
    def test_valid_state_dict_is_constructable(self):
        state: SimulationState = {
            "messages": [HumanMessage(content="Hello")],
            "next_agent": "orchestrator",
            "retrieved_context": ["doc1", "doc2"],
            "memory": {"customer_id": "c1"},
            "session_metadata": {"scenario": "test"},
        }
        assert state["next_agent"] == "orchestrator"
        assert len(state["messages"]) == 1

    def test_next_agent_accepts_known_routes(self):
        for route in ("orchestrator", "sales_exec", "customer", "end"):
            state: SimulationState = {
                "messages": [],
                "next_agent": route,
                "retrieved_context": [],
                "memory": {},
                "session_metadata": {},
            }
            assert state["next_agent"] == route

    def test_retrieved_context_is_list_of_strings(self):
        state: SimulationState = {
            "messages": [],
            "next_agent": "orchestrator",
            "retrieved_context": ["chunk 1", "chunk 2"],
            "memory": {},
            "session_metadata": {},
        }
        assert all(isinstance(c, str) for c in state["retrieved_context"])

    def test_memory_accepts_nested_dict(self):
        state: SimulationState = {
            "messages": [],
            "next_agent": "orchestrator",
            "retrieved_context": [],
            "memory": {"nested": {"key": "value"}},
            "session_metadata": {},
        }
        assert state["memory"]["nested"]["key"] == "value"
