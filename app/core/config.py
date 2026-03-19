from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded securely from the .env file."""
    
    # Feature Toggle
    environment: str = "development"
    llm_provider: str = "azure" # Defaults to azure if missing
    
    # Groq (Optional depending on provider)
    groq_api_key: str | None = None
    groq_model_name: str = "llama3-70b-8192"
    
    # Azure OpenAI (Optional depending on provider)
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_deployment_name: str | None = None
    
    # Database
    mongodb_uri: str = "mongodb://localhost:27017/"
    mongodb_db_name: str = "sales_simulator_db"

    # Opik Observability (Optional so the app doesn't crash if Opik is down)
    opik_url_override: str | None = "http://localhost:5173/api"
    opik_workspace: str = "default"
    opik_project_name: str = "sales-simulator"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()