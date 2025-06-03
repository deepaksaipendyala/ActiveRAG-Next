# graph/nodes.py

from __future__ import annotations

import logging
import os
os.environ["STREAMLIT_WATCH_DIRS"] = "app,graph,loaders,prompts,retrievers,feedback,utils"
import asyncio
import json
import re
from typing import List, Dict, Any, Literal

from graph.state import GraphState, Relation
from retrievers.vectorstore import get_vectorstore
from retrievers.fusion import retrieve_with_fusion
from prompts.coordinator import build_coordinator_prompt
from prompts.reasoner import build_reasoning_prompt
from prompts.validator import build_validation_prompt
from prompts.analyzer import build_analyzer_prompt
from tools.web_search import search_web
from utils.llm import get_llm
from utils.config import settings
from loaders.office_loader import load_office_documents
from loaders.pdf_loader import load_pdf
from loaders.web_loader import load_webpages 

from sentence_transformers import CrossEncoder

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models
llm = get_llm()
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# === NODE FUNCTIONS ===

# --- Coordinator Node ---
def coordinate_workflow(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: coordinate_workflow ---")
    try:
        prompt_template = build_coordinator_prompt(state.query, state.chat_history)
        formatted_prompt = prompt_template.format_prompt().to_string()
        classification = llm.invoke(formatted_prompt)
        query_type = classification.content.strip().lower()
        next_agent = "retriever" if query_type in ["knowledge", "reasoning"] else "retriever"
        return {"query_type": query_type, "next_agent": next_agent, "error": None}
    except Exception as e:
        logger.exception("Coordinator failed.")
        return {"query_type": "unknown", "next_agent": "finalize", "error": f"Coordinator error: {e}"}

def handle_feedback(state: GraphState) -> Dict[str, Any]:
    """
    Looks at the last “[Feedback: N stars] …” system message.
    If rating ≤2, we loop back to reasoner; otherwise we go to finalize.
    """
    logger.info("=== NODE: handle_feedback ===")
    # log entire chat history contents
    logger.info(f"chat_history has {len(state.chat_history or [])} messages:")
    for i, msg in enumerate(state.chat_history or []):
        logger.info(f"  {i}: role={msg.__class__.__name__} content={getattr(msg, 'content', '')!r}")

    # find all feedback‐style system messages
    fb_msgs = [
        m for m in (state.chat_history or [])
        if getattr(m, "content", "").startswith("[Feedback:")
    ]
    logger.info(f"Found {len(fb_msgs)} feedback messages")

    if not fb_msgs:
        logger.info("No feedback → finalize")
        return {
            "use_feedback": False,
            "clarification": None,
            "next_agent": "finalize",
            "error": None
        }

    last = fb_msgs[-1].content
    logger.info(f"Last feedback message: {last!r}")

    m = re.search(r"\[Feedback:\s*(\d)", last)
    rating = int(m.group(1)) if m else 5
    logger.info(f"Parsed rating: {rating}")

    if rating <= 2 and state.retry_count < 2:
        clar = last.split("]", 1)[1].strip()
        logger.info(f"Low rating → loop back to reasoner with clarification={clar!r}")
        return {
            "use_feedback": True,
            "clarification": clar,
            "next_agent": "reasoner",
            "error": None
        }

    logger.info("High rating → finalize")
    return {
        "use_feedback": False,
        "clarification": None,
        "next_agent": "finalize",
        "error": None
    }


# --- Retriever Node ---
async def retrieve_docs(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: retrieve_docs ---")
    query = state.query
    passages: List[str] = []

    try:
        # --- Load local resources ---
        resource_dir = os.path.join(os.getcwd(), "resources")
        office_docs = load_office_documents(resource_dir)
        pdf_docs = []
        for fname in os.listdir(resource_dir):
            if fname.lower().endswith(".pdf"):
                pdf_docs.extend(load_pdf(os.path.join(resource_dir, fname)))

        # --- Load user URLs ---
        urls = state.input_urls or settings.WEB_URLS
        web_docs = load_webpages(urls)

        # --- Populate VectorStore ---
        vectordb = get_vectorstore()
        vectordb.add_documents(office_docs + pdf_docs + web_docs)

        # --- RAG Fusion + Web Search ---
        async def fusion(q):
            return await asyncio.to_thread(retrieve_with_fusion, q, vectordb, llm, num_queries=3, top_n=10)
        async def web_search(q):
            return await asyncio.to_thread(search_web, q, k=3)

        results = await asyncio.gather(fusion(query), web_search(query), return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                passages.extend(res)

        # --- Dedupe & initial ranking ---
        passages = list(dict.fromkeys(passages))
        logger.info(f"Retrieved {len(passages)} passages after fusion and deduplication.")

        pairs = [[query, p] for p in passages]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

        # --- LLM-based Entity Extraction for Boosting ---
        # Sample top-K for entity extraction
        sample_passages = [p for p, _ in ranked][:10]
        analyzer_prompt = build_analyzer_prompt(sample_passages)
        resp = llm.invoke(analyzer_prompt).content
        try:
            payload = json.loads(resp)
            key_entities = {e.lower() for e in payload.get("entities", [])}
        except Exception:
            key_entities = set()

        # --- Boost ranking by entity overlap ---
        boosted = []
        for p, score in ranked:
            boost = sum(1 for ent in key_entities if ent in p.lower())
            boosted.append((p, score + 0.1 * boost))

        boosted = sorted(boosted, key=lambda x: x[1], reverse=True)
        final_passages = [p for p, _ in boosted][:5]

        return {"retrieved_contents": final_passages, "error": None}

    except Exception as e:
        logger.exception("Retriever failed.")
        return {"retrieved_contents": [], "error": f"Retriever error: {e}"}

# --- Analyst Node (Entity & Relation Extraction) ---
async def analyze_docs(state: GraphState) -> Dict[str, Any]:
    """
    Entity & Relation Extraction using a structured‐JSON LLM prompt,
    with resilient JSON parsing and a regex fallback.
    """
    logger.info("--- NODE: analyze_docs ---")
    passages = state.retrieved_contents or []
    if not passages:
        return {
            "extracted_entities": [],
            "extracted_relations": [],
            "knowledge_graph": {},
            "error": None
        }

    raw = ""
    try:
        # 1) Build and invoke the analyzer prompt
        prompt = build_analyzer_prompt(passages)
        raw = llm.invoke(prompt).content

        # 2) Extract first {...} block
        m = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found")
        json_str = m.group(0)

        # 3) Clean control chars & trailing commas
        json_str = re.sub(r"[\x00-\x1f]", " ", json_str)
        json_str = re.sub(r",\s*([\]\}])", r"\1", json_str)

        # 4) Try parse
        payload = json.loads(json_str)
        entities = payload.get("entities", [])
        raw_rels = payload.get("relations", [])

    except Exception as e:
        logger.warning(f"Primary JSON parse failed: {e!r}, falling back to regex")
        # Regex‐fallback: pull out every {...} chunk and extract subject/predicate/object
        entities = set()
        raw_rels = []
        # Find all {...} blocks that look like one relation
        for block in re.finditer(r"\{([^}]+)\}", raw, flags=re.DOTALL):
            text = block.group(1)
            subj_m = re.search(r'"subject"\s*:\s*"([^"]+)"', text)
            pred_m = re.search(r'"predicate"\s*:\s*"([^"]+)"', text)
            obj_m  = re.search(r'"object"\s*:\s*"([^"]+)"', text)
            if subj_m and pred_m and obj_m:
                s = subj_m.group(1).strip()
                p = pred_m.group(1).strip()
                o = obj_m.group(1).strip()
                raw_rels.append({"subject": s, "predicate": p, "object": o})
                entities.update([s, o])

    # 5) Build Relation models
    relations: List[Relation] = []
    for r in raw_rels:
        relations.append(Relation(
            subject   = r["subject"],
            predicate = r["predicate"],
            object    = r["object"]
        ))

    # 6) Build a simple KG dict
    kg: Dict[str, List[str]] = {}
    for rel in relations:
        kg.setdefault(rel.subject, []).append(f"{rel.predicate} → {rel.object}")

    return {
        "extracted_entities": list(entities),
        "extracted_relations": relations,
        "knowledge_graph": kg,
        "error": None
    }

# --- Reasoner Node ---
def reason_through_docs(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: reason_through_docs ---")
    query = state.query
    passages = state.retrieved_contents or []

    clarification = state.clarification.strip() if state.use_feedback and state.clarification else None
    if clarification:
        query = f"{query} (User clarification: {clarification})"

    if not passages:
        return {
            "generation": "No content found.",
            "reasoning_trace": "Empty.",
            "reasoning_steps": [],
            "error": "No passages for reasoning."
        }
    
    try:
        # Rank and summarize top passages
        pairs = [[query, p] for p in passages]
        scores = cross_encoder.predict(pairs)
        # sort passages by cross-encoder score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        top_passages = [passages[i] for i in top_indices]

        summarized = summarize_passages(top_passages)
        context_passages = truncate_passages(summarized)

        reasoning_steps = []
        context = f"Starting Thought: {query}"
        reasoning_steps.append(context)

        for passage in context_passages:
            step_prompt = f"Given: {context}\nNew Info: {passage}\nNext Step:"
            context = llm.invoke(step_prompt).content.strip()
            reasoning_steps.append(context)

        final_prompt = build_reasoning_prompt(query, context_passages)
        final_response = llm.invoke(final_prompt)

        return {
            "generation": final_response.content.strip(),
            "reasoning_trace": "\n\n".join(context_passages),
            "reasoning_steps": reasoning_steps,
            "error": None
        }

    except Exception as e:
        logger.exception("Reasoner failed.")
        return {
            "generation": "Error",
            "reasoning_trace": "Error",
            "reasoning_steps": [],
            "error": f"Reasoner error: {e}"
        }

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
    question   = state.query or ""
    generation = state.generation or ""
    passages   = state.retrieved_contents or []

    # pull in any feedback clarification
    feedback_text = state.clarification if getattr(state, "use_feedback", False) else None

    # If we have nothing to validate against, fail outright
    if not passages or not generation:
        return {
            "is_valid": False,
            "confidence_score": 0.0,
            "validation_feedback": "No generation or no context to validate against.",
            "next_agent": "finalize",
            "retry_count": (state.retry_count or 0) + 1,
            "error": None
        }

    try:
        prompt = build_validation_prompt(
            question=question,
            retrieved_contents=passages,
            generation=generation,
            feedback=feedback_text
        )
        validation = llm.invoke(prompt).content.strip()

        # Expect format "GOOD: reason" or "BAD: reason"
        verdict, _sep, justification = validation.partition(":")
        verdict = verdict.strip().upper()
        justification = justification.strip()

        is_good = verdict == "GOOD"
        score   = 0.9 if is_good else 0.3

        # Default routing
        next_agent = "interactive" if is_good else "reasoner"

        # Cap retries at 2 to break potential cycles
        if (state.retry_count or 0) >= 2:
            next_agent = "finalize"
            logger.warning("Retry limit reached in validator → forcing finalize.")

        return {
            "is_valid": is_good,
            "confidence_score": score,
            "validation_feedback": justification,
            "next_agent": next_agent,
            "retry_count": (state.retry_count or 0) + 1,
            "error": None
        }


    except Exception as e:
        logger.exception("Validator failed.")
        return {
            "is_valid": False,
            "confidence_score": 0.0,
            "validation_feedback": f"Validation error: {e}",
            "next_agent": "finalize",
            "retry_count": (state.retry_count or 0) + 1,
            "error": f"Validator error: {e}"
        }

# --- Interactive Node ---
def interact_with_user(state: GraphState) -> Dict[str, Any]:
    logger.info("--- NODE: interact_with_user ---")
    try:
        # Prepare feedback context
        fb = state.previous_feedback or {}
        rating = fb.get("rating")
        comment = fb.get("comments", "").strip()

        # Modify the prompt based on prior feedback
        feedback_hint = ""
        if rating is not None:
            if rating <= 2:
                feedback_hint = f"\nThe user was unsatisfied with the previous answer. They said: \"{comment}\".\nAdjust your questions to address any lack of clarity or depth."
            elif rating >= 4:
                feedback_hint = f"\nThe user appreciated the previous response. They said: \"{comment}\".\nTry to expand on that or offer deeper insights."

        # Feedback-aware prompts
        clar_prompt = f"Suggest two clarifying questions for:\n{state.query}{feedback_hint}"
        followup_prompt = f"Suggest two follow-up questions for:\n{state.query}{feedback_hint}"

        clarifications = llm.invoke(clar_prompt).content.split("\n")
        followups = llm.invoke(followup_prompt).content.split("\n")

        return {
            "clarification_questions": [q.strip() for q in clarifications if q.strip()],
            "suggested_followups": [q.strip() for q in followups if q.strip()],
            "error": None
        }

    except Exception as e:
        logger.exception("Interactive failed.")
        return {
            "clarification_questions": [],
            "suggested_followups": [],
            "error": f"Interactive error: {e}"
        }

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
    return state.next_agent or "retriever"

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
    if state.retry_count >= 2:
        return "finalize"
    return state.next_agent or "finalize"

