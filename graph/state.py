# graph/state.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class Relation(BaseModel):
    """Represents a relation extracted from documents (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str

class GraphState(BaseModel):
    """
    Shared state object flowing through the LangGraph DAG,
    designed for Multi-Agent Graph-of-Thought architecture.
    """

    # --- Input ---
    query: str
    chat_history: Optional[List[BaseMessage]] = None
    input_urls: List[str] = Field(default_factory=list)

    # --- Coordinator ---
    query_type: Optional[str] = None
    next_agent: Optional[str] = None
    error: Optional[str] = None

    # --- Retriever ---
    documents: List[Document] = Field(default_factory=list)
    retrieved_contents: List[str] = Field(default_factory=list)

    # --- Analyst ---
    extracted_entities: List[str] = Field(default_factory=list)
    extracted_relations: List[Relation] = Field(default_factory=list)
    knowledge_graph: Dict[str, List[str]] = Field(default_factory=dict)

    # --- Reasoner ---
    reasoning_steps: List[str] = Field(default_factory=list)
    reasoning_trace: Optional[str] = None
    generation: Optional[str] = None

    # --- Validator ---
    is_valid: Optional[bool] = None
    validation_feedback: Optional[str] = None
    confidence_score: Optional[float] = None
    retry_count: int = 0

    # --- Interactive / Final Output ---
    clarification_questions: List[str] = Field(default_factory=list)
    suggested_followups: List[str] = Field(default_factory=list)
    answer: Optional[str] = None

    # --- Feedback---
    previous_feedback: Optional[dict] = None
    use_feedback: bool = False
    clarification: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
