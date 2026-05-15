"""Unit tests for app/core/config.py — Settings loading and validation."""
import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from app.core.config import Settings, settings


class TestSettingsDefaults:
    def test_default_llm_provider_is_azure(self):
        s = Settings(_env_file=None)
        assert s.llm_provider == "azure"

    def test_default_environment_is_development(self):
        s = Settings(_env_file=None)
        assert s.environment == "development"

    def test_default_mongodb_uri(self):
        s = Settings(_env_file=None)
        assert s.mongodb_uri == "mongodb://localhost:27017/"

    def test_default_mongodb_db_name(self):
        s = Settings(_env_file=None)
        assert s.mongodb_db_name == "sales_simulator_db"

    def test_default_groq_model(self):
        s = Settings(_env_file=None)
        assert s.groq_model_name == "llama3-70b-8192"

    def test_optional_keys_default_to_none(self):
        s = Settings(_env_file=None)
        assert s.azure_openai_api_key is None
        assert s.azure_openai_endpoint is None
        assert s.groq_api_key is None

    def test_default_opik_workspace(self):
        s = Settings(_env_file=None)
        assert s.opik_workspace == "default"

    def test_default_opik_project_name(self):
        s = Settings(_env_file=None)
        assert s.opik_project_name == "sales-simulator"


class TestSettingsOverride:
    def test_llm_provider_override(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        s = Settings(_env_file=None)
        assert s.llm_provider == "groq"

    def test_groq_api_key_override(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_testkey")
        s = Settings(_env_file=None)
        assert s.groq_api_key == "gsk_testkey"

    def test_mongodb_uri_override(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://remotehost:27017/")
        s = Settings(_env_file=None)
        assert s.mongodb_uri == "mongodb://remotehost:27017/"

    def test_extra_env_vars_are_ignored(self, monkeypatch):
        monkeypatch.setenv("SOME_RANDOM_UNUSED_VAR", "value")
        # Should not raise
        s = Settings(_env_file=None)
        assert not hasattr(s, "some_random_unused_var")


class TestSettingsSingleton:
    def test_module_level_settings_is_settings_instance(self):
        assert isinstance(settings, Settings)

    def test_settings_llm_provider_is_string(self):
        assert isinstance(settings.llm_provider, str)
