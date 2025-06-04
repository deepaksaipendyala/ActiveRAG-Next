# utils/llm.py

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from utils.config import settings

def get_llm():
    """
    Dynamically select the LLM provider (Groq, OpenAI, or Ollama) based on the config settings.
    Returns a LangChain-compatible LLM object, optionally with a 32 K context window.
    """
    provider = settings.LLM_PROVIDER.lower()
    use_32k = getattr(settings, "USE_32K_CONTEXT", False)

    if provider == "groq":
        # Groq’s Llama3 70B supports 8 K by default; if they roll out larger,
        # you can add a branch here.
        model = "llama3-70b-8192"
        return ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model=model,
            temperature=0.2,
            streaming=True,
            max_tokens=4096  # you can bump this to e.g. 8192 if the model supports it
        )

    elif provider == "openai":
        # Switch to the 32 K context variant if requested
        if use_32k:
            model_name = "gpt-4.1-mini-2025-04-14"
            max_toks = 32768
        else:
            model_name = "gpt-4.1-mini-2025-04-14"
            max_toks = 4096

        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=model_name,
            temperature=0.2,
            streaming=True,
            max_tokens=max_toks
        )

    elif provider == "ollama":
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.2,
            streaming=True
        )

    else:
        raise ValueError(f"Unsupported LLM Provider: {provider}")
