"""
Prompt evaluations for the Customer agent.

Uses an LLM-as-judge to score responses on:
  - Persona fidelity: does the customer behave consistently with their role?
  - Objection realism: are objections natural and business-relevant?
  - Resistance to easy persuasion: customer should not agree too quickly?
  - Question quality: does the customer ask probing, intelligent questions?
  - Character consistency: no AI/meta commentary?

Minimum passing score per criterion: 7/10.

Run:
    RUN_EVALS=1 pytest tests/evals/eval_customer.py -v
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from app.prompts.agents.customer import get_customer_prompt
from tests.evals.conftest import llm_judge

PASS_THRESHOLD = 7


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _get_customer_response(eval_llm, messages: list, metadata: dict) -> str:
    prompt_template = get_customer_prompt(session_metadata=metadata)
    formatted = await prompt_template.ainvoke({"messages": messages})
    result = await eval_llm.ainvoke(formatted)
    return result.content


# ---------------------------------------------------------------------------
# Eval: persona fidelity
# ---------------------------------------------------------------------------

@pytest.mark.eval
@pytest.mark.asyncio
async def test_customer_reflects_pain_points_in_questions(
    eval_llm, judge_llm, base_session_metadata
):
    """Customer should ask questions that reflect their stated pain points."""
    messages = [
        AIMessage(content="Hi! I'm Alex from AutoCRM. We help sales teams cut manual admin time by 80%. What's your biggest challenge right now?"),
    ]
    response = await _get_customer_response(eval_llm, messages, base_session_metadata)

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the customer's response reflect their stated pain points "
            "(manual pipeline reporting and low CRM adoption), either by asking "
            "about them or mentioning them as context?"
        ),
        context=f"Customer persona: {base_session_metadata}",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Persona fidelity score {result['score']}/10. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_customer_raises_realistic_objection_on_pricing(
    eval_llm, judge_llm, base_session_metadata
):
    """When pricing is mentioned, the customer should push back realistically."""
    messages = [
        HumanMessage(content="We're evaluating CRM tools for our team."),
        AIMessage(content="Our pricing is $39/seat/month on annual plans — so about $23k/year for 50 seats."),
    ]
    response = await _get_customer_response(eval_llm, messages, base_session_metadata)

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the customer raise a genuine pricing-related concern, question, or objection "
            "rather than immediately accepting the price? Penalise if they agree without pushback."
        ),
        context=f"Customer persona: {base_session_metadata}",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Pricing objection score {result['score']}/10. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_customer_does_not_agree_immediately(
    eval_llm, judge_llm, base_session_metadata
):
    """Customer should require persuasion — not agree to a trial after just one pitch statement."""
    messages = [
        AIMessage(content="AutoCRM will save your reps 5 hours a week and costs half of Salesforce. Want to start a trial?"),
    ]
    response = await _get_customer_response(eval_llm, messages, base_session_metadata)

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the customer resist immediately agreeing to the trial and instead ask "
            "follow-up questions, express scepticism, or request more information? "
            "Penalise if the customer accepts the offer after a single pitch."
        ),
        context=f"Customer persona: {base_session_metadata}",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Resistance score {result['score']}/10. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_customer_asks_probing_questions(
    eval_llm, judge_llm, base_session_metadata
):
    """Customer should ask at least one specific, intelligent question per turn."""
    messages = [
        AIMessage(content="We integrate natively with Slack, Gmail, and Zoom and can have you onboarded in two weeks."),
    ]
    response = await _get_customer_response(eval_llm, messages, base_session_metadata)

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the customer ask at least one specific, probing question "
            "(e.g., about ROI, security, migration, or team adoption)? "
            "Penalise for vague or no questions."
        ),
        context=f"Customer persona: {base_session_metadata}",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Question quality score {result['score']}/10. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_customer_stays_in_character(eval_llm, judge_llm, base_session_metadata):
    """Customer must not break character or reveal AI nature."""
    messages = [
        AIMessage(content="What does your current CRM reporting workflow look like?"),
    ]
    response = await _get_customer_response(eval_llm, messages, base_session_metadata)

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the customer stay in character as a real business buyer throughout? "
            "Penalise if they reveal they are an AI, reference the simulation, "
            "or use non-human phrasing."
        ),
        context="The customer must role-play as a human business buyer at all times.",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Character consistency score {result['score']}/10. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_customer_persona_varies_by_metadata(eval_llm, judge_llm):
    """Different personas should produce noticeably different response styles."""
    cfo_metadata = {
        "buyer_persona": "CFO at a 200-person enterprise",
        "buyer_pain_points": "ROI visibility and budget justification for software spend",
        "scenario": "Finance-led CRM evaluation",
    }
    messages = [
        AIMessage(content="Our platform gives your sales team full pipeline visibility. Ready to start a trial?"),
    ]

    response = await _get_customer_response(eval_llm, messages, cfo_metadata)

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the response sound like a CFO focused on ROI and budget justification "
            "rather than a generic buyer? Penalise if it could be any buyer persona."
        ),
        context=f"Customer persona: {cfo_metadata}",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Persona specificity score {result['score']}/10. Reason: {result['reason']}\n"
        f"Response: {response}"
    )
