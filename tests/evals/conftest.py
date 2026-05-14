"""
Shared fixtures and utilities for prompt evaluation tests.

Evals use a real LLM (configured via .env) and an LLM-as-judge to score
outputs against defined rubrics. They are marked with @pytest.mark.eval
and skipped in CI unless RUN_EVALS=1 is set.

Run evals:
    RUN_EVALS=1 pytest tests/evals/ -v
"""
import os
import json
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Skip guard — evals are opt-in
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    if not os.getenv("RUN_EVALS"):
        skip = pytest.mark.skip(reason="Set RUN_EVALS=1 to run prompt evaluations")
        for item in items:
            if item.get_closest_marker("eval"):
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# LLM fixture (real, from settings)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_llm():
    """Real LLM for generating agent responses during evals."""
    from app.core.llm import get_llm
    return get_llm(temperature=0.1)


@pytest.fixture(scope="session")
def judge_llm():
    """Separate LLM instance used as the evaluator/judge."""
    from app.core.llm import get_llm
    return get_llm(temperature=0.0)


# ---------------------------------------------------------------------------
# Judge utility
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert evaluator assessing AI agent responses in a B2B sales simulation.
Score the response on the given criterion from 0 to 10.
Respond ONLY with a JSON object: {{"score": <int 0-10>, "reason": "<one sentence>"}}"""


async def llm_judge(judge_llm, criterion: str, context: str, response: str) -> dict:
    """
    Asks the judge LLM to score `response` on `criterion` given `context`.
    Returns {"score": int, "reason": str}.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM),
        ("human", "Criterion: {criterion}\n\nContext:\n{context}\n\nResponse to evaluate:\n{response}"),
    ])
    chain = prompt | judge_llm
    result = await chain.ainvoke(
        {"criterion": criterion, "context": context, "response": response}
    )
    try:
        return json.loads(result.content)
    except (json.JSONDecodeError, AttributeError):
        return {"score": -1, "reason": "Judge failed to return valid JSON."}


# ---------------------------------------------------------------------------
# Shared scenario data
# ---------------------------------------------------------------------------

@pytest.fixture
def crm_conversation():
    """A mid-conversation CRM sales exchange used across multiple eval tests."""
    return [
        HumanMessage(content="Hi, we're evaluating CRM tools for our 50-person sales team."),
        AIMessage(content="Great! What's your biggest challenge with your current setup?"),
        HumanMessage(content="Our reps spend too much time on manual data entry and pipeline reporting."),
    ]


@pytest.fixture
def objection_conversation():
    """Conversation with a tough pricing objection."""
    return [
        HumanMessage(content="We already use Salesforce. It's expensive but the whole team knows it."),
        AIMessage(content="That's completely understandable. Switching costs are real. What would make it worth evaluating an alternative?"),
        HumanMessage(content="Honestly, if it saved my reps 5+ hours a week and cost less than half of Salesforce."),
    ]


@pytest.fixture
def base_session_metadata():
    return {
        "scenario": "Enterprise CRM pitch",
        "buyer_persona": "VP of Sales at a 50-person SaaS company",
        "buyer_pain_points": "manual pipeline reporting and low CRM adoption",
    }


@pytest.fixture
def product_context():
    return [
        "AutoCRM reduces manual data entry by 80% through AI-powered call transcription.",
        "Pricing: $39/seat/month (annual), $49/seat/month (monthly). Volume discounts at 25+ seats.",
        "Integrates natively with Slack, Gmail, Outlook, Zoom, and 200+ tools via Zapier.",
        "Average onboarding time: 2 weeks with a dedicated Customer Success Manager.",
    ]
