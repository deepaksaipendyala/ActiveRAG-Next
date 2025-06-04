import os
import pytest

# Minimal environment configuration so `utils.config.Settings` can load
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
os.environ.setdefault("EMBEDDING_MODEL", "dummy")

from graph.builder import run_active_rag
from graph.state import GraphState


@pytest.mark.asyncio
async def test_active_rag_simple_query():
    query = "What is the capital of France?"
    final_state = await run_active_rag(query=query)

    assert isinstance(final_state, GraphState)
    assert final_state.answer
    assert final_state.is_valid is not None


@pytest.mark.asyncio
async def test_active_rag_reasoning_query():
    query = "Why is the sky blue during the day but turns red at sunset?"
    final_state = await run_active_rag(query=query)

    assert isinstance(final_state, GraphState)
    assert final_state.answer
    assert final_state.is_valid is not None


@pytest.mark.asyncio
async def test_active_rag_validation_loop():
    query = (
        "If I drop a ball and an elephant from a building, which falls faster "
        "according to Aristotle?"
    )
    final_state = await run_active_rag(query=query)

    assert isinstance(final_state, GraphState)
    assert final_state.answer
    assert final_state.is_valid is not None
