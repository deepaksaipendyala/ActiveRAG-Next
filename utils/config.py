# utils/config.py

from pydantic_settings import BaseSettings
from typing import Optional
from typing import List

class Settings(BaseSettings):
    # LLM settings
    LLM_PROVIDER: str
    GROQ_API_KEY: str
    OPENAI_API_KEY: str

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

    class Config:
        env_file = ".env"

settings = Settings()
