# graph/builder.py

import logging
from typing import List
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    coordinate_workflow,
    handle_feedback,
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

# ─── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Graph Construction ────────────────────────────────────────
def _build_graph() -> StateGraph:
    logger.info("Building ActiveRAG LangGraph…")
    g = StateGraph(GraphState)

    # Register nodes
    g.add_node("coordinator", coordinate_workflow)
    g.add_node("retriever",   retrieve_docs)
    g.add_node("analyst",     analyze_docs)
    g.add_node("reasoner",    reason_through_docs)
    g.add_node("validator",   validate_reasoning)
    g.add_node("interactive", interact_with_user)
    g.add_node("feedback",    handle_feedback)
    g.add_node("finalize",    finalize)

    # Entry point
    g.set_entry_point("coordinator")

    # Conditional routing
    g.add_conditional_edges(
        "coordinator", route_query,
        {
            "retriever": "retriever",
            "reasoner":  "reasoner",
            "validator": "validator",
            "finalize":  "finalize",
            "__end__":   END,
        }
    )
    g.add_conditional_edges(
        "retriever", decide_after_retrieval,
        {
            "analyst":  "analyst",
            "finalize": "finalize",
        }
    )
    g.add_conditional_edges(
        "analyst", decide_after_analysis,
        {
            "reasoner":  "reasoner",
            "finalize":  "finalize",
        }
    )
    g.add_conditional_edges(
        "reasoner", decide_after_reasoning,
        {
            "validator": "validator",
            "finalize":  "finalize",
        }
    )
    g.add_conditional_edges(
        "validator", route_after_validation,
        {
            "interactive": "interactive",
            "reasoner":    "reasoner",
            "finalize":    "finalize",
            "__end__":     END,
        }
    )

    # Final arcs
    # After interactive, always run feedback handler
    g.add_edge("interactive", "feedback")
    # feedback → either loop back to reasoner, forward to finalize, or (if it hands back “interactive”) go straight to finalize
    g.add_conditional_edges(
        "feedback",
        # if next_agent is None (i.e. no retry), we want "finalize"
        lambda state: state.next_agent or "finalize",
        {
            "reasoner":    "reasoner",
            "finalize":    "finalize",
            "interactive": "finalize",  # stale interactive also goes to finalize
            "__end__":     END,
        }
    )
    # If feedback decides no further work, finalize ends the graph
    g.add_edge("finalize",    END)


    # Compile
    logger.info("Compiling ActiveRAG graph…")
    compiled = g.compile()
    logger.info("Graph compiled successfully.")
    return compiled

# Compile once
_compiled_graph = _build_graph()

# ─── Public API ────────────────────────────────────────────────
async def run_active_rag(
    query: str,
    chat_history: list = None,
    input_urls: List[str] = None,
    stream: bool = False
) -> GraphState:
    """
    Runs the full ActiveRAG pipeline end-to-end.
    Returns the final GraphState.
    """
    logger.info(f"Running ActiveRAG for query: {query!r}")
    init = GraphState(
        query=query,
        chat_history=chat_history or [],
        input_urls=input_urls or []
    )
    # Async invoke through all nodes
    result = await _compiled_graph.ainvoke(init, config={"recursion_limit": 20})
    # If returned dict, pydantic will coerce; otherwise already a GraphState
    return result if isinstance(result, GraphState) else GraphState(**result)


async def stream_active_rag(
    query: str,
    chat_history: list = None,
    input_urls: List[str] = None
):
    """
    Streams outputs as each node completes.
    Yields (node_name, output_dict) for each step.
    """
    logger.info(f"Streaming ActiveRAG for query: {query!r}")
    init = GraphState(
        query=query,
        chat_history=chat_history or [],
        input_urls=input_urls or []
    )
    async for chunk in _compiled_graph.astream(init, config={"recursion_limit": 20}):
        node, out = next(iter(chunk.items()))
        yield node, out
    logger.info("Streaming complete.")


def get_graphviz_dot() -> str:
    compiled = _compiled_graph.get_graph()  # networkx graph
    lines = ["digraph ActiveRAG {"]
    for edge in compiled.edges:
        src, dst = edge[0], edge[1]
        lines.append(f'    "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)
