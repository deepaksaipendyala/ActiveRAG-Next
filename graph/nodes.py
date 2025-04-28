# graph/nodes.py

from __future__ import annotations

import logging
import asyncio
from typing import List, Dict, Any, Literal

from graph.state import GraphState, Relation
from retrievers.vectorstore import get_vectorstore
from retrievers.fusion import retrieve_with_fusion
from prompts.coordinator import build_coordinator_prompt
from prompts.reasoner import build_reasoning_prompt
from prompts.validator import build_validation_prompt
from tools.web_search import search_web
from utils.llm import get_llm
from utils.config import settings

from sentence_transformers import CrossEncoder
import spacy

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models
llm = get_llm()
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
ner_model = spacy.load("en_core_web_sm")

# === NODE FUNCTIONS ===

# --- Coordinator Node ---
def coordinate_workflow(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: coordinate_workflow ---")
    try:
        prompt_template = build_coordinator_prompt(state.query, state.chat_history)
        formatted_prompt = prompt_template.format_prompt().to_string()  # 🛠 Format the prompt properly
        classification = llm.invoke(formatted_prompt)
        query_type = classification.content.strip().lower()
        next_agent = "retriever" if query_type in ["knowledge", "reasoning"] else "retriever"
        return {"query_type": query_type, "next_agent": next_agent, "error": None}
    except Exception as e:
        logger.exception("Coordinator failed.")
        return {"query_type": "unknown", "next_agent": "finalize", "error": f"Coordinator error: {e}"}


# --- Retriever Node ---
async def retrieve_docs(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: retrieve_docs ---")
    query = state.query
    passages = []

    try:
        vectordb = get_vectorstore()

        async def retrieve_with_fusion_async(q):
            return await asyncio.to_thread(retrieve_with_fusion, q, vectordb, llm, num_queries=3, top_n=5)

        async def web_search_async(q):
            return await asyncio.to_thread(search_web, q, k=3)

        tasks = [
            retrieve_with_fusion_async(query),
            web_search_async(query)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, list):
                passages.extend(res)

        unique_passages = list(dict.fromkeys(passages))
        logger.info(f"Retrieved {len(unique_passages)} unique passages.")
        return {"retrieved_contents": unique_passages, "error": None}

    except Exception as e:
        logger.exception("Retriever failed.")
        return {"retrieved_contents": [], "error": f"Retriever error: {e}"}


# --- Analyst Node ---
def analyze_docs(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: analyze_docs ---")
    contents = state.retrieved_contents or []
    entities, relations, kg = set(), [], {}

    if not contents:
        logger.warning("No content to analyze.")
        return {"extracted_entities": [], "extracted_relations": [], "knowledge_graph": {}, "error": None}

    try:
        for passage in contents:
            doc = ner_model(passage[:ner_model.max_length])
            for ent in doc.ents:
                entities.add(ent.text.strip())

            relation_prompt = f"Extract subject, predicate, object triples:\n\n{passage}"
            response = llm.invoke(relation_prompt)
            triples = response.content.split("\n")
            for triple in triples:
                parts = triple.split(",")
                if len(parts) == 3:
                    relations.append(Relation(subject=parts[0].strip(), predicate=parts[1].strip(), object=parts[2].strip()))

        for rel in relations:
            kg.setdefault(rel.subject, []).append(f"{rel.predicate} -> {rel.object}")

        return {"extracted_entities": sorted(list(entities)), "extracted_relations": relations, "knowledge_graph": kg, "error": None}

    except Exception as e:
        logger.exception("Analyst failed.")
        return {"extracted_entities": [], "extracted_relations": [], "knowledge_graph": {}, "error": f"Analyst error: {e}"}

# --- Reasoner Node ---
def reason_through_docs(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: reason_through_docs ---")
    query = state.query
    passages = state.retrieved_contents or []

    if not passages:
        return {"generation": "No content found.", "reasoning_trace": "Empty.", "reasoning_steps": [], "error": "No passages for reasoning."}

    try:
        pairs = [[query, p] for p in passages]
        scores = cross_encoder.predict(pairs)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        top_passages = [passages[i] for i in top_indices]

        # Summarize and truncate
        summarized = summarize_passages(top_passages)
        context_passages = truncate_passages(summarized)

        # Graph of Thought
        reasoning_steps = []
        context = f"Starting Thought: {query}"
        reasoning_steps.append(context)

        for passage in context_passages:
            step_prompt = f"Given: {context}\nNew Info: {passage}\nNext Step:"
            context = llm.invoke(step_prompt).content.strip()
            reasoning_steps.append(context)

        final_prompt = build_reasoning_prompt(query, context_passages)
        final_response = llm.invoke(final_prompt)

        return {"generation": final_response.content.strip(), "reasoning_trace": "\n\n".join(context_passages), "reasoning_steps": reasoning_steps, "error": None}

    except Exception as e:
        logger.exception("Reasoner failed.")
        return {"generation": "Error", "reasoning_trace": "Error", "reasoning_steps": [], "error": f"Reasoner error: {e}"}

def summarize_passages(passages: List[str], batch_size: int = 5) -> List[str]:
    summaries = []
    batch = []
    for passage in passages:
        batch.append(passage)
        if len(batch) >= batch_size:
            joined = "\n\n".join(batch)
            summary = llm.invoke(f"Summarize:\n{joined}")
            summaries.append(summary.content.strip())
            batch = []
    if batch:
        joined = "\n\n".join(batch)
        summary = llm.invoke(f"Summarize:\n{joined}")
        summaries.append(summary.content.strip())
    return summaries

def truncate_passages(passages: List[str], max_chars: int = 7000) -> List[str]:
    total, output = 0, []
    for passage in passages:
        if total + len(passage) > max_chars:
            break
        output.append(passage)
        total += len(passage)
    return output

# --- Validator Node ---
def validate_reasoning(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: validate_reasoning ---")
    query = state.query
    generation = state.generation

    if not generation:
        return {"is_valid": False, "confidence_score": 0.0, "validation_feedback": "Missing generation.", "next_agent": "finalize", "error": "No generation"}

    try:
        prompt = build_validation_prompt(query, generation)
        validation = llm.invoke(prompt).content.strip()

        if validation.upper().startswith("GOOD"):
            return {"is_valid": True, "confidence_score": 0.85, "validation_feedback": validation, "next_agent": "interactive", "error": None}
        else:
            return {"is_valid": False, "confidence_score": 0.5, "validation_feedback": validation, "next_agent": "reasoner", "error": None}

    except Exception as e:
        logger.exception("Validator failed.")
        return {"is_valid": False, "confidence_score": 0.0, "validation_feedback": "Validation error", "next_agent": "finalize", "error": f"Validator error: {e}"}

# --- Interactive Node ---
def interact_with_user(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: interact_with_user ---")
    try:
        clarifications = llm.invoke(f"Suggest two clarifying questions for:\n{state.query}").content.split("\n")
        followups = llm.invoke(f"Suggest two follow-up questions for:\n{state.query}").content.split("\n")
        return {"clarification_questions": clarifications, "suggested_followups": followups, "error": None}
    except Exception as e:
        logger.exception("Interactive failed.")
        return {"clarification_questions": [], "suggested_followups": [], "error": f"Interactive error: {e}"}

# --- Finalizer Node ---
def finalize(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: finalize ---")
    if state.error:
        return {"answer": f"Error: {state.error}"}
    if state.is_valid:
        return {"answer": state.generation}
    else:
        return {"answer": f"Validation failed. Feedback: {state.validation_feedback or 'No feedback'}"}



# === CONDITIONAL FUNCTIONS ===

def route_query(state: GraphState) -> Literal["retriever", "reasoner", "validator", "finalize", "__end__"]:
    return state.next_agent or 'retriever'

def decide_after_retrieval(state: GraphState) -> Literal["analyst", "finalize"]:
    if not state.retrieved_contents:
        return "finalize"
    return "analyst"

def decide_after_analysis(state: GraphState) -> Literal["reasoner", "finalize"]:
    return "reasoner"

def decide_after_reasoning(state: GraphState) -> Literal["validator", "finalize"]:
    if not state.generation:
        return "finalize"
    return "validator"

def route_after_validation(state: GraphState) -> Literal["interactive", "reasoner", "finalize", "__end__"]:
    return state.next_agent or 'finalize'
