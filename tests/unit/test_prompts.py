"""Unit tests for app/prompts/agents/ — prompt generation functions."""
import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.agents.orchestrator import get_orchestrator_prompt
from app.prompts.agents.sales_exec import get_sales_exec_prompt
from app.prompts.agents.customer import get_customer_prompt


class TestOrchestratorPrompt:
    def test_returns_chat_prompt_template(self):
        result = get_orchestrator_prompt()
        assert isinstance(result, ChatPromptTemplate)

    def test_contains_messages_placeholder(self):
        result = get_orchestrator_prompt()
        variable_names = [m.variable_name for m in result.messages if hasattr(m, "variable_name")]
        assert "messages" in variable_names

    def test_default_scenario_in_system_message(self):
        result = get_orchestrator_prompt()
        system_content = result.messages[0].prompt.template
        assert "general B2B sales call" in system_content

    def test_custom_scenario_injected(self):
        metadata = {"scenario": "Enterprise cybersecurity pitch"}
        result = get_orchestrator_prompt(session_metadata=metadata)
        system_content = result.messages[0].prompt.template
        assert "Enterprise cybersecurity pitch" in system_content

    def test_none_metadata_uses_defaults(self):
        result = get_orchestrator_prompt(session_metadata=None)
        system_content = result.messages[0].prompt.template
        assert "general B2B sales call" in system_content

    def test_routing_instructions_present(self):
        result = get_orchestrator_prompt()
        system_content = result.messages[0].prompt.template
        assert "sales_exec" in system_content
        assert "customer" in system_content
        assert "end" in system_content


class TestSalesExecPrompt:
    def test_returns_chat_prompt_template(self):
        result = get_sales_exec_prompt(retrieved_context=[], memory={})
        assert isinstance(result, ChatPromptTemplate)

    def test_context_injected_into_system_message(self):
        context = ["Product supports SSO.", "Free 30-day trial."]
        result = get_sales_exec_prompt(retrieved_context=context, memory={})
        system_content = result.messages[0].prompt.template
        assert "Product supports SSO." in system_content
        assert "Free 30-day trial." in system_content

    def test_empty_context_uses_fallback(self):
        result = get_sales_exec_prompt(retrieved_context=[], memory={})
        system_content = result.messages[0].prompt.template
        assert "No specific product context available." in system_content

    def test_memory_injected_into_system_message(self):
        memory = {"past_objections": ["price"], "industry": "FinTech"}
        result = get_sales_exec_prompt(retrieved_context=[], memory=memory)
        system_content = result.messages[0].prompt.template
        assert "FinTech" in system_content

    def test_empty_memory_uses_fallback(self):
        result = get_sales_exec_prompt(retrieved_context=[], memory=None)
        system_content = result.messages[0].prompt.template
        assert "No prior interaction history" in system_content

    def test_contains_messages_placeholder(self):
        result = get_sales_exec_prompt(retrieved_context=[], memory={})
        variable_names = [m.variable_name for m in result.messages if hasattr(m, "variable_name")]
        assert "messages" in variable_names

    def test_product_knowledge_tag_present(self):
        result = get_sales_exec_prompt(retrieved_context=["some context"], memory={})
        system_content = result.messages[0].prompt.template
        assert "<product_knowledge>" in system_content

    def test_customer_history_tag_present(self):
        result = get_sales_exec_prompt(retrieved_context=[], memory={"key": "val"})
        system_content = result.messages[0].prompt.template
        assert "<customer_history>" in system_content


class TestCustomerPrompt:
    def test_returns_chat_prompt_template(self):
        result = get_customer_prompt()
        assert isinstance(result, ChatPromptTemplate)

    def test_default_persona_used_when_no_metadata(self):
        result = get_customer_prompt(session_metadata=None)
        system_content = result.messages[0].prompt.template
        assert "pragmatic B2B software buyer" in system_content

    def test_custom_persona_injected(self):
        metadata = {"buyer_persona": "CFO at a mid-market logistics company"}
        result = get_customer_prompt(session_metadata=metadata)
        system_content = result.messages[0].prompt.template
        assert "CFO at a mid-market logistics company" in system_content

    def test_custom_pain_points_injected(self):
        metadata = {"buyer_pain_points": "lack of real-time inventory visibility"}
        result = get_customer_prompt(session_metadata=metadata)
        system_content = result.messages[0].prompt.template
        assert "lack of real-time inventory visibility" in system_content

    def test_default_pain_points_used_when_absent(self):
        metadata = {"buyer_persona": "VP Sales"}  # no pain_points key
        result = get_customer_prompt(session_metadata=metadata)
        system_content = result.messages[0].prompt.template
        assert "budget constraints" in system_content

    def test_contains_messages_placeholder(self):
        result = get_customer_prompt()
        variable_names = [m.variable_name for m in result.messages if hasattr(m, "variable_name")]
        assert "messages" in variable_names

    def test_objection_instruction_present(self):
        result = get_customer_prompt()
        system_content = result.messages[0].prompt.template
        assert "objection" in system_content.lower()
