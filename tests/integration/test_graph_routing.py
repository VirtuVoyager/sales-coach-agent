"""
Integration tests for the LangGraph orchestration graph.

Tests the graph compilation, node wiring, and routing logic without
making real LLM or database calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from app.agents.orchestrator.router import AgentRoute, RouteDecision
from app.agents.state import SimulationState


@pytest.fixture
def mock_checkpointer():
    cp = MagicMock()
    cp.aget_tuple = AsyncMock(return_value=None)
    cp.aput = AsyncMock()
    cp.aput_writes = AsyncMock()
    return cp


@pytest.fixture
def mock_llm_sales_customer():
    """LLM that returns a simple AIMessage for sales/customer nodes."""
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content="Mock agent response.")
    return llm


@pytest.fixture
def mock_llm_orchestrator_end():
    """Orchestrator LLM that immediately routes to END."""
    llm = AsyncMock()
    structured = AsyncMock()
    structured.ainvoke.return_value = RouteDecision(next_agent=AgentRoute.END_ROUTE)
    llm.with_structured_output.return_value = structured
    return llm


@pytest.fixture
def mock_llm_orchestrator_sales():
    """Orchestrator LLM that routes to sales_exec once, then END."""
    llm = AsyncMock()
    structured = AsyncMock()
    structured.ainvoke.side_effect = [
        RouteDecision(next_agent=AgentRoute.SALES_EXEC),
        RouteDecision(next_agent=AgentRoute.END_ROUTE),
    ]
    llm.with_structured_output.return_value = structured
    return llm


class TestGraphCompilation:
    def test_graph_compiles_without_error(self, mock_checkpointer):
        from app.agents.orchestrator.graph import SimulationOrchestrator
        with patch("app.agents.orchestrator.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock()
            mock_get_llm.return_value = mock_llm
            orchestrator = SimulationOrchestrator(checkpointer=mock_checkpointer)
        assert orchestrator.graph is not None

    def test_graph_has_orchestrator_node(self, mock_checkpointer):
        from app.agents.orchestrator.graph import SimulationOrchestrator
        with patch("app.agents.orchestrator.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock()
            mock_get_llm.return_value = mock_llm
            orchestrator = SimulationOrchestrator(checkpointer=mock_checkpointer)
        assert "orchestrator" in orchestrator.graph.nodes

    def test_graph_has_sales_exec_node(self, mock_checkpointer):
        from app.agents.orchestrator.graph import SimulationOrchestrator
        with patch("app.agents.orchestrator.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock()
            mock_get_llm.return_value = mock_llm
            orchestrator = SimulationOrchestrator(checkpointer=mock_checkpointer)
        assert "sales_exec" in orchestrator.graph.nodes

    def test_graph_has_customer_node(self, mock_checkpointer):
        from app.agents.orchestrator.graph import SimulationOrchestrator
        with patch("app.agents.orchestrator.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock()
            mock_get_llm.return_value = mock_llm
            orchestrator = SimulationOrchestrator(checkpointer=mock_checkpointer)
        assert "customer" in orchestrator.graph.nodes


class TestGraphRouting:
    @pytest.mark.asyncio
    async def test_orchestrator_routes_to_end_immediately(
        self, mock_checkpointer, mock_llm_orchestrator_end, base_state
    ):
        from app.agents.orchestrator.graph import SimulationOrchestrator
        with patch("app.agents.orchestrator.graph.get_llm", return_value=mock_llm_orchestrator_end), \
             patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            orchestrator = SimulationOrchestrator(checkpointer=mock_checkpointer)
            config = {"configurable": {"thread_id": "test-thread-001"}}
            result = await orchestrator.ainvoke(base_state, config)

        assert result["next_agent"] == "end"

    @pytest.mark.asyncio
    async def test_orchestrator_routes_through_sales_exec(
        self,
        mock_checkpointer,
        mock_llm_orchestrator_sales,
        mock_llm_sales_customer,
        base_state,
    ):
        from app.agents.orchestrator.graph import SimulationOrchestrator
        from app.agents.sales_exec.node import SalesExecNode
        from app.agents.customer.node import CustomerNode

        with patch("app.agents.orchestrator.graph.get_llm", return_value=mock_llm_orchestrator_sales), \
             patch("app.agents.orchestrator.graph.SalesExecNode") as MockSales, \
             patch("app.agents.orchestrator.graph.CustomerNode"), \
             patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_orch_prompt, \
             patch("app.agents.sales_exec.node.get_sales_exec_prompt") as mock_sales_prompt:

            # Mock SalesExec to return a Command going back to orchestrator
            from langgraph.types import Command
            sales_instance = AsyncMock()
            sales_instance.ainvoke.return_value = Command(
                update={"messages": [AIMessage(content="Here is our pricing.")]},
                goto="orchestrator",
            )
            MockSales.return_value = sales_instance

            mock_orch_prompt.return_value = AsyncMock()
            mock_orch_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            mock_sales_prompt.return_value = AsyncMock()
            mock_sales_prompt.return_value.ainvoke = AsyncMock(return_value=[])

            orchestrator = SimulationOrchestrator(checkpointer=mock_checkpointer)
            config = {"configurable": {"thread_id": "test-thread-002"}}
            result = await orchestrator.ainvoke(base_state, config)

        assert result["next_agent"] == "end"

    @pytest.mark.asyncio
    async def test_thread_id_config_required(self, mock_checkpointer, mock_llm_orchestrator_end, base_state):
        """Graph should work when thread_id is provided."""
        from app.agents.orchestrator.graph import SimulationOrchestrator
        with patch("app.agents.orchestrator.graph.get_llm", return_value=mock_llm_orchestrator_end), \
             patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            orchestrator = SimulationOrchestrator(checkpointer=mock_checkpointer)
            config = {"configurable": {"thread_id": "unique-thread-123"}}
            result = await orchestrator.ainvoke(base_state, config)

        assert "messages" in result


class TestHealthEndpoint:
    def test_health_check_returns_healthy(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch("app.main.MongoClient"))
            stack.enter_context(patch("app.main.load_vector_store", return_value=None))
            stack.enter_context(patch("certifi.where", return_value="/mock/ca-bundle.crt"))

            from fastapi.testclient import TestClient
            from app.main import app
            with TestClient(app) as client:
                resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
