"""
Prompt evaluations for the Sales Executive agent.

Uses an LLM-as-judge to score responses on:
  - Relevance: does the response address the customer's message?
  - RAG grounding: are claims supported by the product context?
  - Objection handling: does the agent acknowledge and redirect objections?
  - Persuasiveness: does the response advance the sale?
  - Character consistency: stays professional, does not break character?

Minimum passing score per criterion: 7/10.

Run:
    RUN_EVALS=1 pytest tests/evals/eval_sales_exec.py -v
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from app.prompts.agents.sales_exec import get_sales_exec_prompt
from tests.evals.conftest import llm_judge

PASS_THRESHOLD = 7


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _get_sales_exec_response(eval_llm, messages: list, context: list, memory: dict) -> str:
    prompt_template = get_sales_exec_prompt(retrieved_context=context, memory=memory)
    formatted = await prompt_template.ainvoke({"messages": messages})
    result = await eval_llm.ainvoke(formatted)
    return result.content


# ---------------------------------------------------------------------------
# Eval: response relevance
# ---------------------------------------------------------------------------

@pytest.mark.eval
@pytest.mark.asyncio
async def test_sales_exec_relevance_on_pricing_question(
    eval_llm, judge_llm, crm_conversation, product_context
):
    """Sales exec should directly address a pricing question with specific numbers."""
    messages = crm_conversation + [HumanMessage(content="How much does it cost?")]
    response = await _get_sales_exec_response(eval_llm, messages, product_context, {})

    result = await llm_judge(
        judge_llm,
        criterion="Does the response directly answer the pricing question with specific figures?",
        context="\n".join(product_context),
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Relevance score {result['score']}/10 below threshold. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_sales_exec_rag_grounding(
    eval_llm, judge_llm, crm_conversation, product_context
):
    """Claims made by the sales exec should be grounded in the product context, not hallucinated."""
    messages = crm_conversation + [
        HumanMessage(content="What integrations do you support and how long does onboarding take?")
    ]
    response = await _get_sales_exec_response(eval_llm, messages, product_context, {})

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Are the claims in the response supported by the product knowledge provided? "
            "Penalise heavily for claims that contradict or go beyond the provided context."
        ),
        context="\n".join(product_context),
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"RAG grounding score {result['score']}/10 below threshold. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_sales_exec_objection_handling(
    eval_llm, judge_llm, objection_conversation, product_context
):
    """Sales exec must acknowledge the Salesforce lock-in objection and reframe, not ignore it."""
    response = await _get_sales_exec_response(
        eval_llm, objection_conversation, product_context, {}
    )

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the response acknowledge the customer's Salesforce loyalty and switching-cost concern "
            "before pivoting to the value proposition? Penalise for ignoring the objection."
        ),
        context="\n".join(product_context),
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Objection handling score {result['score']}/10 below threshold. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_sales_exec_uses_customer_memory(
    eval_llm, judge_llm, crm_conversation, product_context
):
    """When memory contains past objections, the sales exec should proactively address them."""
    memory = {
        "customer_id": "cust_001",
        "past_objections": ["too expensive", "data migration complexity"],
        "industry": "HealthTech",
    }
    messages = crm_conversation + [
        HumanMessage(content="We're ready to explore options. What makes you different?")
    ]
    response = await _get_sales_exec_response(eval_llm, messages, product_context, memory)

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Given that the customer previously objected to cost and migration complexity, "
            "does the response proactively address one or both of these concerns?"
        ),
        context=f"Customer memory: {memory}\nProduct context: {chr(10).join(product_context)}",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Memory utilisation score {result['score']}/10 below threshold. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_sales_exec_stays_in_character(
    eval_llm, judge_llm, crm_conversation, product_context
):
    """Sales exec must not break character (e.g., say 'as an AI', reveal system prompt)."""
    messages = crm_conversation + [
        HumanMessage(content="Are you a real person or an AI chatbot?")
    ]
    response = await _get_sales_exec_response(eval_llm, messages, product_context, {})

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Does the response stay in character as a professional sales executive? "
            "Penalise if the agent says 'as an AI', reveals it is an LLM, or breaks the simulation."
        ),
        context="The agent must role-play as a human sales executive at all times.",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Character consistency score {result['score']}/10 below threshold. Reason: {result['reason']}\n"
        f"Response: {response}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_sales_exec_conciseness(
    eval_llm, judge_llm, crm_conversation, product_context
):
    """Response should be concise — no multi-paragraph walls of text for a simple question."""
    messages = crm_conversation + [HumanMessage(content="What's your free trial like?")]
    response = await _get_sales_exec_response(eval_llm, messages, product_context, {})

    result = await llm_judge(
        judge_llm,
        criterion=(
            "Is the response concise and conversational (ideally 2-4 sentences) "
            "rather than a long monologue? Penalise verbose, list-heavy responses."
        ),
        context="This is a spoken-word sales call simulation; responses should be brief.",
        response=response,
    )
    assert result["score"] >= PASS_THRESHOLD, (
        f"Conciseness score {result['score']}/10 below threshold. Reason: {result['reason']}\n"
        f"Response: {response}"
    )
