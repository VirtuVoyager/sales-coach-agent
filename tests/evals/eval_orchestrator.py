"""
Prompt evaluations for the Orchestrator — routing correctness.

Evals verify that the orchestrator routes to the correct agent
given a specific conversation state. No LLM judge needed here:
routing is deterministic enough to assert directly.

Run:
    RUN_EVALS=1 pytest tests/evals/eval_orchestrator.py -v
"""
import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from app.agents.orchestrator.router import AgentRoute, RouteDecision, OrchestratorNode
from app.agents.state import SimulationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(messages: list, metadata: dict | None = None) -> SimulationState:
    return {
        "messages": messages,
        "next_agent": "orchestrator",
        "retrieved_context": [],
        "memory": {},
        "session_metadata": metadata or {},
    }


async def _route(eval_llm, messages: list, metadata: dict | None = None) -> str:
    """
    Runs the orchestrator with a real LLM and returns the routing decision string.
    """
    structured_llm = eval_llm.with_structured_output(RouteDecision)
    node = OrchestratorNode.__new__(OrchestratorNode)
    node.name = "orchestrator"
    node.llm = structured_llm

    state = _state(messages, metadata)
    from app.prompts.agents.orchestrator import get_orchestrator_prompt
    prompt_template = get_orchestrator_prompt(state.get("session_metadata"))
    formatted = await prompt_template.ainvoke({"messages": state["messages"]})
    decision: RouteDecision = await structured_llm.ainvoke(formatted)
    return decision.next_agent.value


# ---------------------------------------------------------------------------
# Routing correctness evals
# ---------------------------------------------------------------------------

@pytest.mark.eval
@pytest.mark.asyncio
async def test_routes_to_end_after_ai_speaks(eval_llm, base_session_metadata):
    """After the AI sales exec speaks, orchestrator must pause for human reply → 'end'."""
    messages = [
        HumanMessage(content="Tell me about your pricing."),
        AIMessage(content="Our pricing starts at $39/seat/month on annual plans."),
    ]
    route = await _route(eval_llm, messages, base_session_metadata)
    assert route == "end", f"Expected 'end' after AI spoke, got '{route}'"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_routes_to_sales_exec_after_human_speaks(eval_llm, base_session_metadata):
    """After the human customer speaks, orchestrator must route to 'sales_exec'."""
    messages = [
        HumanMessage(content="Hi, I'm evaluating CRM tools for my team."),
    ]
    route = await _route(eval_llm, messages, base_session_metadata)
    assert route == "sales_exec", f"Expected 'sales_exec', got '{route}'"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_routes_to_sales_exec_after_customer_objection(eval_llm, base_session_metadata):
    """Customer objection should trigger the sales exec to respond."""
    messages = [
        HumanMessage(content="We use Salesforce and the team loves it. Why would we switch?"),
        AIMessage(content="Great question — what does your team love most about Salesforce?"),
        HumanMessage(content="The pipeline reporting. But it costs us $150/seat/month."),
    ]
    route = await _route(eval_llm, messages, base_session_metadata)
    assert route == "sales_exec", f"Expected 'sales_exec' after customer objection, got '{route}'"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_routes_to_end_when_deal_concluded(eval_llm, base_session_metadata):
    """When the conversation reaches a natural conclusion, orchestrator should end."""
    messages = [
        HumanMessage(content="This sounds great. Let's set up a trial."),
        AIMessage(content="Wonderful! I'll send you the onboarding link right away. Looking forward to working with you!"),
    ]
    route = await _route(eval_llm, messages, base_session_metadata)
    assert route == "end", f"Expected 'end' after deal conclusion, got '{route}'"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_routes_to_customer_in_automated_mode(eval_llm):
    """In automated mode (no human actor), orchestrator should alternate to customer after sales exec."""
    metadata = {
        "scenario": "Automated agent-to-agent demo",
        "mode": "automated",
    }
    messages = [
        AIMessage(content="Hello! I'm Alex from AutoCRM. How can I help your sales team today?"),
    ]
    route = await _route(eval_llm, messages, metadata)
    assert route == "customer", f"Expected 'customer' in automated mode, got '{route}'"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_routing_valid_enum_value(eval_llm, base_session_metadata):
    """Orchestrator must always return a valid AgentRoute value — never an arbitrary string."""
    messages = [HumanMessage(content="What integrations do you support?")]
    route = await _route(eval_llm, messages, base_session_metadata)
    valid_routes = {r.value for r in AgentRoute}
    assert route in valid_routes, f"'{route}' is not a valid AgentRoute value"


# ---------------------------------------------------------------------------
# Prompt quality: does the system prompt produce coherent output?
# ---------------------------------------------------------------------------

@pytest.mark.eval
@pytest.mark.asyncio
async def test_scenario_context_affects_routing_prompt(eval_llm):
    """Different scenarios should still produce valid routing decisions."""
    scenarios = [
        "Cybersecurity SaaS pitch to a CISO",
        "HR analytics demo to a People Ops leader",
        "Supply chain software for a logistics VP",
    ]
    messages = [HumanMessage(content="Tell me more about your solution.")]
    for scenario in scenarios:
        route = await _route(eval_llm, messages, {"scenario": scenario})
        valid = {r.value for r in AgentRoute}
        assert route in valid, f"Scenario '{scenario}' produced invalid route: '{route}'"
