# utils/config.py

from pydantic_settings import BaseSettings
from typing import Optional, List
from dotenv import load_dotenv  # NEW

# Load environment variables first
load_dotenv(override=True)  # <-- ADD THIS before initializing Settings()

class Settings(BaseSettings):

    # LLM settings
    LLM_PROVIDER: str
    GROQ_API_KEY: str
    OPENAI_API_KEY: str
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # Embedding settings
    EMBEDDING_PROVIDER: str
    EMBEDDING_MODEL: str

    # Vectorstore settings
    CHROMA_DB_PATH: str = "./chroma_db"

    # Web Search / RAG-Fusion settings
    TAVILY_API_KEY: Optional[str] = None

    PDF_DIRECTORY: str = "/Users/deepaksaipendyala/Documents/MultiAgent/resources"

    OFFICE_DIRECTORY: str = "./office"
    WEB_URLS: List[str] = []

    use_32k_context: bool = False

    class Config:
        env_file = ".env"

# Initialize settings after dotenv is loaded
settings = Settings()
