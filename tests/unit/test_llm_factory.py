"""Unit tests for app/core/llm.py — LLM provider factory."""
import pytest
from unittest.mock import patch, MagicMock

from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq

from app.core.llm import get_llm


AZURE_ENV = {
    "LLM_PROVIDER": "azure",
    "AZURE_OPENAI_API_KEY": "test-azure-key",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o",
}

GROQ_ENV = {
    "LLM_PROVIDER": "groq",
    "GROQ_API_KEY": "gsk_testkey",
}


class TestGetLLMAzure:
    def test_returns_azure_chat_openai_instance(self, monkeypatch):
        for k, v in AZURE_ENV.items():
            monkeypatch.setenv(k, v)
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "azure"
            mock_settings.azure_openai_api_key = "test-azure-key"
            mock_settings.azure_openai_endpoint = "https://test.openai.azure.com"
            mock_settings.azure_openai_api_version = "2024-02-15-preview"
            mock_settings.azure_openai_deployment_name = "gpt-4o"
            llm = get_llm()
        assert isinstance(llm, AzureChatOpenAI)

    def test_azure_missing_key_raises(self):
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "azure"
            mock_settings.azure_openai_api_key = None
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
                get_llm()

    def test_azure_temperature_passed_through(self):
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "azure"
            mock_settings.azure_openai_api_key = "key"
            mock_settings.azure_openai_endpoint = "https://test.openai.azure.com"
            mock_settings.azure_openai_api_version = "2024-02-15"
            mock_settings.azure_openai_deployment_name = "gpt-4o"
            llm = get_llm(temperature=0.7)
        assert llm.temperature == 0.7


class TestGetLLMGroq:
    def test_returns_chat_groq_instance(self):
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "groq"
            mock_settings.groq_api_key = "gsk_testkey"
            mock_settings.groq_model_name = "llama3-70b-8192"
            llm = get_llm()
        assert isinstance(llm, ChatGroq)

    def test_groq_missing_key_raises(self):
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "groq"
            mock_settings.groq_api_key = None
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                get_llm()

    def test_groq_model_name_propagated(self):
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "groq"
            mock_settings.groq_api_key = "gsk_testkey"
            mock_settings.groq_model_name = "mixtral-8x7b-32768"
            llm = get_llm()
        assert llm.model_name == "mixtral-8x7b-32768"


class TestGetLLMUnknownProvider:
    def test_unknown_provider_raises_value_error(self):
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "anthropic"
            with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
                get_llm()

    def test_empty_provider_raises(self):
        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.llm_provider = ""
            with pytest.raises(ValueError):
                get_llm()
