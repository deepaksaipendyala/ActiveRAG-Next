# test_active_rag.py
import sys
import os
import logging
import asyncio

# Allow imports if running from project root
sys.path.append(os.path.abspath("."))

from graph.builder import run_active_rag
from graph.state import GraphState

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# === Test 1: Simple factual query ===
def test_active_rag_simple_query():
    logger.info("=== Test 1: Simple factual query ===")
    query = "What is the capital of France?"
    final_state = asyncio.run(run_active_rag(query=query))

    assert isinstance(final_state, GraphState), "Final output is not a GraphState instance"
    assert final_state.answer, "Final answer is empty!"

    print("\n=== FINAL ANSWER ===")
    print(final_state.answer)
    print("\n=== Reasoning Trace ===")
    print(final_state.reasoning_trace or "No reasoning trace available.")
    print("\n=== Validation Feedback ===")
    print(final_state.validation_feedback or "No validation feedback available.")

# === Test 2: Reasoning-heavy query ===
def test_active_rag_reasoning_query():
    logger.info("=== Test 2: Reasoning-heavy query ===")
    query = "Why is the sky blue during the day but turns red at sunset?"
    final_state = asyncio.run(run_active_rag(query=query))

    assert isinstance(final_state, GraphState), "Final output is not a GraphState instance"
    assert final_state.answer, "Final answer is empty!"

    print("\n=== FINAL ANSWER ===")
    print(final_state.answer)
    print("\n=== Reasoning Trace ===")
    print(final_state.reasoning_trace or "No reasoning trace available.")
    print("\n=== Validation Feedback ===")
    print(final_state.validation_feedback or "No validation feedback available.")

# === Test 3: Validator-loop triggered query ===
def test_active_rag_validation_loop():
    logger.info("=== Test 3: Validator-loop tricky query ===")
    query = "If I drop a ball and an elephant from a building, which falls faster according to Aristotle?"
    final_state = asyncio.run(run_active_rag(query=query))

    assert isinstance(final_state, GraphState), "Final output is not a GraphState instance"
    assert final_state.answer, "Final answer is empty!"

    print("\n=== FINAL ANSWER ===")
    print(final_state.answer)
    print("\n=== Reasoning Trace ===")
    print(final_state.reasoning_trace or "No reasoning trace available.")
    print("\n=== Validation Feedback ===")
    print(final_state.validation_feedback or "No validation feedback available.")
    print("\n=== Reasoning Steps ===")
    for idx, step in enumerate(final_state.reasoning_steps or []):
        print(f"Step {idx+1}: {step}")

# --- Execute tests selectively ---
if __name__ == "__main__":
    # Choose which tests to run
    test_active_rag_simple_query()
    print("\n" + "="*80 + "\n")
    # test_active_rag_reasoning_query()
    # print("\n" + "="*80 + "\n")
    # test_active_rag_validation_loop()
