from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq

from app.core.config import settings

def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """
    Instantiates and returns the configured LLM based on the .env toggle.
    """
    provider = settings.llm_provider.lower()
    
    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Check your .env file.")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
            temperature=temperature
        )
        
    elif provider == "azure":
        if not settings.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is missing. Check your .env file.")
        return AzureChatOpenAI(
            azure_deployment=settings.azure_openai_deployment_name,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            temperature=temperature
        )
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Please use 'azure' or 'groq'.")