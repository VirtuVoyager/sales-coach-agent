"""Unit tests for app/agents/sales_exec/node.py — SalesExecNode."""
import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.agents.sales_exec.node import SalesExecNode


class TestSalesExecNode:
    @pytest.fixture
    def node(self, mock_llm):
        return SalesExecNode("sales_exec", mock_llm)

    @pytest.mark.asyncio
    async def test_returns_command(self, node, base_state):
        with patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert isinstance(result, Command)

    @pytest.mark.asyncio
    async def test_goto_is_orchestrator(self, node, base_state):
        with patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert result.goto == "orchestrator"

    @pytest.mark.asyncio
    async def test_update_contains_messages(self, node, base_state):
        with patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert "messages" in result.update
        assert len(result.update["messages"]) == 1
        assert isinstance(result.update["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_llm_response_appended(self, node, base_state, mock_llm):
        expected_content = "Our CRM cuts rep admin time by 40%."
        mock_llm.ainvoke.return_value = AIMessage(content=expected_content)

        with patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(base_state)

        assert result.update["messages"][0].content == expected_content

    @pytest.mark.asyncio
    async def test_prompt_receives_rag_context(self, node, base_state):
        with patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            await node.ainvoke(base_state)

        call_kwargs = mock_prompt.call_args
        assert call_kwargs.kwargs["retrieved_context"] == base_state["retrieved_context"]

    @pytest.mark.asyncio
    async def test_prompt_receives_memory(self, node, base_state):
        with patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            await node.ainvoke(base_state)

        call_kwargs = mock_prompt.call_args
        assert call_kwargs.kwargs["memory"] == base_state["memory"]

    @pytest.mark.asyncio
    async def test_empty_context_and_memory_handled(self, node, empty_state):
        with patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            result = await node.ainvoke(empty_state)

        assert isinstance(result, Command)
        assert result.goto == "orchestrator"

    def test_node_name_stored(self, node):
        assert node.name == "sales_exec"
