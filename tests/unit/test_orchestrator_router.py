"""Unit tests for app/agents/orchestrator/router.py — routing logic and schema."""
import pytest
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError
from langgraph.graph import END

from app.agents.orchestrator.router import AgentRoute, RouteDecision, OrchestratorNode


class TestAgentRouteEnum:
    def test_sales_exec_value(self):
        assert AgentRoute.SALES_EXEC == "sales_exec"

    def test_customer_value(self):
        assert AgentRoute.CUSTOMER == "customer"

    def test_end_route_value(self):
        assert AgentRoute.END_ROUTE == "end"

    def test_enum_is_str_subclass(self):
        assert isinstance(AgentRoute.SALES_EXEC, str)

    def test_all_three_routes_exist(self):
        values = {r.value for r in AgentRoute}
        assert values == {"sales_exec", "customer", "end"}


class TestRouteDecision:
    def test_valid_sales_exec_decision(self):
        d = RouteDecision(next_agent=AgentRoute.SALES_EXEC)
        assert d.next_agent == AgentRoute.SALES_EXEC

    def test_valid_customer_decision(self):
        d = RouteDecision(next_agent=AgentRoute.CUSTOMER)
        assert d.next_agent == AgentRoute.CUSTOMER

    def test_valid_end_decision(self):
        d = RouteDecision(next_agent=AgentRoute.END_ROUTE)
        assert d.next_agent == AgentRoute.END_ROUTE

    def test_invalid_route_raises(self):
        with pytest.raises(ValidationError):
            RouteDecision(next_agent="unknown_agent")

    def test_missing_next_agent_raises(self):
        with pytest.raises(ValidationError):
            RouteDecision()


class TestOrchestratorNode:
    @pytest.fixture
    def mock_structured_llm(self):
        llm = AsyncMock()
        llm.with_structured_output.return_value = llm
        return llm

    @pytest.fixture
    def orchestrator(self, mock_structured_llm):
        return OrchestratorNode("orchestrator", mock_structured_llm)

    @pytest.mark.asyncio
    async def test_routes_to_sales_exec(self, orchestrator, base_state):
        decision = RouteDecision(next_agent=AgentRoute.SALES_EXEC)
        orchestrator.llm.ainvoke.return_value = decision

        with patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            command = await orchestrator.ainvoke(base_state)

        assert command.goto == "sales_exec"
        assert command.update["next_agent"] == "sales_exec"

    @pytest.mark.asyncio
    async def test_routes_to_customer(self, orchestrator, base_state):
        decision = RouteDecision(next_agent=AgentRoute.CUSTOMER)
        orchestrator.llm.ainvoke.return_value = decision

        with patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            command = await orchestrator.ainvoke(base_state)

        assert command.goto == "customer"
        assert command.update["next_agent"] == "customer"

    @pytest.mark.asyncio
    async def test_routes_to_end_constant(self, orchestrator, base_state):
        """When routing to 'end', goto must be LangGraph's END constant, not the string 'end'."""
        decision = RouteDecision(next_agent=AgentRoute.END_ROUTE)
        orchestrator.llm.ainvoke.return_value = decision

        with patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            command = await orchestrator.ainvoke(base_state)

        assert command.goto == END
        assert command.update["next_agent"] == "end"

    @pytest.mark.asyncio
    async def test_state_metadata_passed_to_prompt(self, orchestrator, base_state):
        decision = RouteDecision(next_agent=AgentRoute.END_ROUTE)
        orchestrator.llm.ainvoke.return_value = decision

        with patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            await orchestrator.ainvoke(base_state)

        mock_prompt.assert_called_once_with(base_state["session_metadata"])

    @pytest.mark.asyncio
    async def test_empty_state_does_not_crash(self, orchestrator, empty_state):
        decision = RouteDecision(next_agent=AgentRoute.END_ROUTE)
        orchestrator.llm.ainvoke.return_value = decision

        with patch("app.agents.orchestrator.router.get_orchestrator_prompt") as mock_prompt:
            mock_prompt.return_value = AsyncMock()
            mock_prompt.return_value.ainvoke = AsyncMock(return_value=[])
            command = await orchestrator.ainvoke(empty_state)

        assert command.goto == END
