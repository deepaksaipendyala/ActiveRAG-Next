# utils/llm.py

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from utils.config import settings

def get_llm():
    """
    Dynamically select the LLM provider (Groq or OpenAI) based on the config settings.
    Returns a LangChain compatible LLM object.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq":
        return ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="llama3-70b-8192",   # or llama3-8b-8192
            temperature=0.2,
            streaming=True,
            max_tokens=4096
        )

    elif provider == "openai":
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.2,
            streaming=True,
            max_tokens=4096
        )

    else:
        raise ValueError(f"Unsupported LLM Provider: {provider}")
