# graph/builder.py
import logging
import asyncio
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    coordinate_workflow,
    retrieve_docs,
    analyze_docs,
    reason_through_docs,
    validate_reasoning,
    interact_with_user,
    finalize,
    route_query,
    decide_after_retrieval,
    decide_after_analysis,
    decide_after_reasoning,
    route_after_validation,
)
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger(__name__)

def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph DAG for the 6-Agent ActiveRAG architecture,
    incorporating conditional logic for dynamic routing and error handling.
    """
    logger.info("Building the Multi-Agent ActiveRAG graph with conditional logic...")

    workflow = StateGraph(GraphState)

    # --- Add all nodes (Agents) ---
    workflow.add_node("coordinator", coordinate_workflow)
    workflow.add_node("retriever", retrieve_docs)
    workflow.add_node("analyst", analyze_docs)
    workflow.add_node("reasoner", reason_through_docs)
    workflow.add_node("validator", validate_reasoning)
    workflow.add_node("interactive", interact_with_user)
    workflow.add_node("finalize", finalize)

    # --- Define the dynamic edges ---
    workflow.set_entry_point("coordinator")

    workflow.add_conditional_edges(
        "coordinator",
        route_query,
        {
            "retriever": "retriever",
            "reasoner": "reasoner",
            "validator": "validator",
            "finalize": "finalize",
            "__end__": END,
        }
    )

    workflow.add_conditional_edges(
        "retriever",
        decide_after_retrieval,
        {
            "analyst": "analyst",
            "finalize": "finalize",
        }
    )

    workflow.add_conditional_edges(
        "analyst",
        decide_after_analysis,
        {
            "reasoner": "reasoner",
            "finalize": "finalize",
        }
    )

    workflow.add_conditional_edges(
        "reasoner",
        decide_after_reasoning,
        {
            "validator": "validator",
            "finalize": "finalize",
        }
    )

    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "interactive": "interactive",
            "reasoner": "reasoner",
            "finalize": "finalize",
            "__end__": END,
        }
    )

    workflow.add_edge("interactive", "finalize")
    workflow.add_edge("finalize", END)

    logger.info("Compiling the graph...")
    compiled_graph = workflow.compile()
    logger.info("Graph compiled successfully.")

    return compiled_graph

async def run_active_rag(query: str, chat_history: List = None) -> GraphState:
    """
    Async version of ActiveRAG runner.
    """
    compiled_graph = build_graph()
    initial_state = GraphState(query=query, chat_history=chat_history or [])
    logger.info(f"\n--- Running ActiveRAG for Query: '{query}' ---")
    
    # AWAIT async invoke now
    final_state_dict = await compiled_graph.ainvoke(initial_state, config={"recursion_limit": 10})
    logger.info("--- RAG Run Finished ---")

    # Convert dict to GraphState
    if isinstance(final_state_dict, dict):
        try:
            final_state = GraphState(**final_state_dict)
            return final_state
        except Exception as e:
            logger.error(f"Failed to parse final state dictionary back to GraphState: {e}")
            return final_state_dict
    elif isinstance(final_state_dict, GraphState):
        return final_state_dict
    else:
        logger.error(f"Unexpected final state type: {type(final_state_dict)}")
        return final_state_dict

# Example streaming (optional)
# def stream_active_rag(query: str, chat_history: List = None):
#     compiled_graph = build_graph()
#     initial_state = GraphState(query=query, chat_history=chat_history or [])
#     logger.info(f"\n--- Streaming ActiveRAG for Query: '{query}' ---")
#     final_snapshot = None
#     for chunk in compiled_graph.stream(initial_state, config={"recursion_limit": 10}):
#         node_name = list(chunk.keys())[0]
#         node_output = chunk[node_name]
#         logger.info(f"Output from '{node_name}': {node_output}")
#         final_snapshot = node_output
#     logger.info("--- RAG Streaming Finished ---")
#     return final_snapshot
