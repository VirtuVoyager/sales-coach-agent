"""Unit tests for app/agents/customer/node.py — CustomerNode."""
import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.agents.customer.node import CustomerNode


class TestCustomerNode:
    @pytest.fixture
    def node(self, mock_llm):
        return CustomerNode("customer", mock_llm)

    @pytest.mark.asyncio
    async def test_returns_command(self, node, base_state):
        with patch("app.agents.customer.node.get_customer_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert isinstance(result, Command)

    @pytest.mark.asyncio
    async def test_goto_is_orchestrator(self, node, base_state):
        with patch("app.agents.customer.node.get_customer_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert result.goto == "orchestrator"

    @pytest.mark.asyncio
    async def test_update_contains_ai_message(self, node, base_state):
        with patch("app.agents.customer.node.get_customer_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert "messages" in result.update
        assert isinstance(result.update["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_prompt_receives_session_metadata(self, node, base_state):
        with patch("app.agents.customer.node.get_customer_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            await node.ainvoke(base_state)

        mock_prompt.assert_called_once_with(session_metadata=base_state["session_metadata"])

    @pytest.mark.asyncio
    async def test_persona_reflected_in_prompt_call(self, node, base_state):
        """Metadata with specific buyer_persona must be forwarded to get_customer_prompt."""
        base_state["session_metadata"]["buyer_persona"] = "CFO at a Fortune 500 company"

        with patch("app.agents.customer.node.get_customer_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            await node.ainvoke(base_state)

        call_kwargs = mock_prompt.call_args
        assert call_kwargs.kwargs["session_metadata"]["buyer_persona"] == "CFO at a Fortune 500 company"

    @pytest.mark.asyncio
    async def test_none_metadata_handled(self, node, empty_state):
        with patch("app.agents.customer.node.get_customer_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(empty_state)

        assert result.goto == "orchestrator"

    @pytest.mark.asyncio
    async def test_llm_response_propagated(self, node, base_state, mock_llm):
        objection = "We already use Salesforce. Why should we switch?"
        mock_llm.ainvoke.return_value = AIMessage(content=objection)

        with patch("app.agents.customer.node.get_customer_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert result.update["messages"][0].content == objection

    def test_node_name_stored(self, node):
        assert node.name == "customer"
