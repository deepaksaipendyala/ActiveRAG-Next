import os
os.environ["TORCH_DISABLE_STREAMLIT_WATCH"] = "1"

import streamlit as st
import sys
import re
# Refined watch dirs/exclude if needed based on your exact project structure
os.environ["STREAMLIT_WATCH_DIRS"] = "app,graph,loaders,prompts,retrievers,feedback,utils"
# os.environ["STREAMLIT_WATCH_EXCLUDE_PATTERNS"] = r"\/torch\/|\/.venv\/" # Example broader exclude
import asyncio
import time
import logging
from pydantic import BaseModel
import json
from typing import Dict, List

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from graph.builder import run_active_rag, get_graphviz_dot, stream_active_rag
from graph.state import GraphState, Relation # Import the state model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Serialization Helper ---
def to_serializable(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json') # Use mode='json' for better serialization
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    # Handle other non-serializable types if necessary
    try:
        json.dumps(obj) # Test if serializable
        return obj
    except TypeError:
        return str(obj) # Convert problematic types to string as fallback

# --- Chat History Conversion ---
def convert_chat_to_base_messages(chat_list):
    messages = []
    for m in chat_list:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
        elif m["role"] == "system":
            messages.append(SystemMessage(content=m.get("content", "")))
    return messages

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - UI - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger

# --- Page Configuration ---
st.set_page_config(
    page_title="ActiveRAG Next",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def display_state_details(state: GraphState | None): # Allow None state
    """Helper to display various details from the final state in tabs."""
    if not state:
        st.warning("Graph execution did not yield a final state.")
        return
    if not isinstance(state, GraphState):
        st.error(f"Invalid state type received: {type(state)}. Cannot display details.")
        logger.error(f"Display function received invalid state type: {type(state)}")
        return

    # Use tabs for better organization
    tab_answer, tab_reasoning, tab_context, tab_analysis, tab_interactive, tab_flow = st.tabs([
        "Final Answer",
        "Reasoning",
        "Context",
        "Analysis",
        "Interaction",
        "Execution Flow"
    ])

    # --- Tab 1: Final Answer ---
    with tab_answer:
        st.subheader("Final Answer")
        # Use getattr for safety, although state should conform to GraphState
        final_answer = getattr(state, 'answer', None)
        if final_answer:
            st.markdown(final_answer) # Display the final formatted answer
        else:
            st.warning("No final answer was generated in the state.")
            # Optionally display validation feedback if answer is missing but validation failed
            if not getattr(state, 'is_valid', True) and getattr(state, 'validation_feedback', None):
                 st.info(f"Validation Feedback: {state.validation_feedback}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            is_valid = getattr(state, 'is_valid', None)
            validation_text = "N/A"
            if is_valid is True:
                validation_text = "✅ Valid"
            elif is_valid is False:
                validation_text = "⚠️ Invalid"
            st.metric(label="Validation Status", value=validation_text)
        with col2:
            score = getattr(state, 'confidence_score', None)
            st.metric(label="Confidence Score", value=f"{score:.2f}" if score is not None else "N/A")

        val_feedback = getattr(state, 'validation_feedback', None)
        if val_feedback and is_valid is False:
             with st.expander("Show Validation Feedback"):
                  st.warning(val_feedback)

    # --- Tab 2: Reasoning ---
    with tab_reasoning:
        st.subheader("Reasoner Output (Final Generation)")
        generation = getattr(state, 'generation', None)
        if generation:
            # Use markdown in case the generation includes formatting
            st.markdown("```\n" + generation + "\n```")
            # st.text_area("Raw Generation Output", generation, height=200) # Alternative display
        else:
            st.info("No raw generation output available in the final state.")

        st.subheader("Reasoning Steps (Trace)")
        steps = getattr(state, 'reasoning_steps', [])
        if steps:
            # Using markdown with code block for better readability
            st.markdown("```markdown\n" + "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps)) + "\n```")
        else:
            st.info("No detailed reasoning steps available in the final state.")

        st.subheader("Final Context Used for Reasoning (Trace)")
        trace = getattr(state, 'reasoning_trace', None)
        if trace:
            # Use markdown in case trace includes formatting, wrap in code block
             st.markdown("```\n" + trace + "\n```")
            # st.text_area("Context Trace", trace, height=300) # Alternative
        else:
            st.info("No reasoning context trace available in the final state.")

    # --- Tab 3: Context ---
    with tab_context:
        st.subheader("Retrieved Content")
        contents = getattr(state, 'retrieved_contents', [])
        if contents:
            st.info(f"{len(contents)} passages retrieved (showing top passages used/available).")
            # Decide whether to show all or just top ones used in reasoning_trace if available
            with st.expander("View All Retrieved Passages", expanded=False):
                for idx, passage in enumerate(contents):
                    st.markdown(f"**Passage {idx+1}:**")
                    # Use st.markdown or st.text_area based on expected content length/type
                    st.markdown(f"```text\n{passage}\n```")
                    # st.text_area(f"passage_{idx+1}", passage, height=150, label_visibility="collapsed", key=f"ctx_{idx}")
                    st.markdown("---")
        else:
            st.warning("No content was retrieved or retained in the final state.")
        # Add display for input URLs if implemented and stored in state
        input_urls = getattr(state, 'input_urls', []) # Assuming you add input_urls to GraphState
        if input_urls:
           st.subheader("User Provided URLs")
           st.write(input_urls)

    # --- Tab 4: Analysis ---
    with tab_analysis:
        st.subheader("Analyst Output")
        entities = getattr(state, 'extracted_entities', [])
        relations = getattr(state, 'extracted_relations', [])

        # show counts + raw triples
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entities Extracted", len(entities))
            if entities:
                with st.expander("View Entities"):
                    st.write(entities)
        with col2:
            st.metric("Relations Extracted", len(relations))
            if relations:
                with st.expander("View Relations (Triples)"):
                    st.json([rel.model_dump() for rel in relations])

        # --- Build a focused Knowledge Graph ---
        # --- Knowledge‑Graph ----------------------------------------------------------
        st.subheader("📈 Knowledge Graph")

        if not relations:
            st.info("No relations found – nothing to draw.")
        else:
            raw_q = (state.query or "").strip()

            # a. pattern‐based extraction from the user query
            patterns = [
                r'(?i)^\s*who\s+is\s+(.*)',
                r'(?i)^\s*what\s+is\s+(.*)',
                r'(?i)^\s*tell\s+me\s+about\s+(.*)',
                r'(?i)^\s*give\s+me\s+information\s+about\s+(.*)',
                r'(?i)^\s*explain\s+(?:the\s+)?(.*)',
                r'(?i)^\s*describe\s+(.*)',
            ]
            centre = None
            for pat in patterns:
                m = re.match(pat, raw_q)
                if m:
                    centre = m.group(1).rstrip("?. ").strip()
                    break

            # b. fallback – most common subject in the triples
            if not centre:
                from collections import Counter
                subj_counts = Counter(r.subject for r in relations if r.subject)
                if subj_counts:
                    centre, _ = subj_counts.most_common(1)[0]

            if not centre:
                st.info("Couldn’t guess a central entity.")
            else:
                # Optional UI – let the user switch to a different subject
                unique_subjects = sorted({r.subject for r in relations})

                # Try to match centre in a case-insensitive way
                match_index = next(
                    (i for i, subj in enumerate(unique_subjects) if subj.lower() == centre.lower()),
                    0  # fallback to first subject if no match
                )
                chosen = st.selectbox("Focus node:", unique_subjects, index=match_index)
                centre = chosen

                def clean(txt: str) -> str:
                    return (txt.replace("*", "")
                            .replace("(", "").replace(")", "")
                            .replace('"', r'\"')
                            .strip())

                c_lbl = clean(centre)

                dot = [
                    "digraph KG {",
                    "  rankdir=LR;",
                    '  node [style=filled,fontname="Helvetica"];',
                    f'  "{c_lbl}" [shape=oval, fillcolor=lightgoldenrod1, fontsize=12];',
                ]

                seen = set()
                for r in relations:
                    if r.subject.lower() != centre.lower():
                        continue                   # only edges out of the centre node

                    s, p, o = clean(r.subject), clean(r.predicate), clean(r.object)
                    if (s, p, o) in seen:
                        continue                  # de‑duplicate
                    seen.add((s, p, o))

                    dot.append(f'  "{o}" [shape=box, fillcolor=whitesmoke, fontsize=10];')
                    dot.append(f'  "{s}" -> "{o}" [label="{p}", fontcolor=gray40];')

                dot.append("}")
                dot_src = "\n".join(dot)

                st.graphviz_chart(dot_src)
                with st.expander("View DOT source"):
                    st.code(dot_src, language="dot")

    # --- Tab 5: Interaction ---
    with tab_interactive:
        st.subheader("Interactive Agent Output")
        clarifications = getattr(state, 'clarification_questions', [])
        followups = getattr(state, 'suggested_followups', [])

        if clarifications:
            st.markdown("**Suggested Clarification Questions:**")
            # Filter out potential empty strings or placeholder text from LLM output
            filtered_clarifications = [q for q in clarifications if q and not q.startswith("Here are")]
            if filtered_clarifications:
                for q in filtered_clarifications:
                    st.markdown(f"- {q}")
            else:
                 st.info("No specific clarification questions generated.")
        else:
            st.info("No clarification questions generated.")

        st.divider()

        if followups:
            st.markdown("**Suggested Follow-up Questions:**")
            # Filter out potential empty strings or placeholder text
            filtered_followups = [q for q in followups if q and not q.startswith("Here are")]
            if filtered_followups:
                for q in filtered_followups:
                    st.markdown(f"- {q}")
            else:
                st.info("No specific follow-up questions generated.")
        else:
            st.info("No follow-up questions generated.")

    # --- Tab 6: Execution Flow ---
    with tab_flow:
        st.subheader("Agent Execution Flow")
        # This requires capturing the flow during graph.stream()
        if 'execution_flow' in st.session_state and st.session_state.execution_flow:
            st.markdown(" -> ".join(f"`{node}`" for node in st.session_state.execution_flow))
            # Optionally display state changes at each step if captured and stored
        else:
            st.info("Execution flow tracking not available or graph run via invoke.")

        # Placeholder for graph visualization
        try:
            st.subheader("📈 StateGraph DAG")
            dot = get_graphviz_dot()
            st.graphviz_chart(dot)
        except Exception as viz_e:
            st.warning(f"Could not generate graph visualization: {viz_e}")

# --- Main Application Logic ---
def main():
    st.title("ActiveRAG Next - Multi-Agent Reasoning Demo")
    st.markdown("""
    Ask any question and see the multi-agent system retrieve information,
    analyze it, reason through it, validate the results, and generate interactive suggestions.
    Configure options in the sidebar.
    """)

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model Selection (Example)
        # TODO: Populate with actual available models from config/detection
        available_llms = ["Groq Llama3", "OpenAI GPT-4o", "Ollama Model"] # Example list
        selected_llm = st.selectbox("Select LLM:", available_llms, index=0)

        # Web References Input
        st.subheader("Add Web References (Optional)")
        input_urls_str = st.text_area(
            "Enter URLs (one per line):",
            height=100,
            placeholder="e.g., https://example.com/article\nhttps://another-site.org/info"
            # Need to modify the 'retrieve_docs' node to fetch and process these.
        )
        input_urls = [url.strip() for url in input_urls_str.split("\n") if url.strip()]

        # Document Set Selection (Placeholder)
        st.subheader("Select Document Context (Placeholder)")
        # TODO: Implement logic to list and select available document sets/indices
        st.info("Document set selection (VectorDB) is handled automatically by the retriever.")
        # selected_docs = st.multiselect("Use Documents:", ["Set A", "Set B"], default=["Set A"]) # Example

        st.divider()
        # run_button = st.button("Run ActiveRAG", use_container_width=True) # Button removed, using chat_input trigger

    # --- Main Area Logic ---
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Initialize execution flow tracking
    if "execution_flow" not in st.session_state:
        st.session_state.execution_flow = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if user_query := st.chat_input("🔍 Ask your question here..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Prepare arguments for the backend runner
        # Currently, the graph nodes are self-contained and use global settings.
        rag_config = {
            "llm_provider_ui_selection": selected_llm, # Pass selection if needed by backend
            "input_urls": input_urls, # Pass URLs if retriever node uses them
        }

        # --- Run the Graph ---
        final_state = None
        st.session_state.execution_flow = []  # Reset flow for new query

        # accumulate partial outputs into one dict
        state_updates = {"query": user_query}
        current_execution_flow = []

        with st.chat_message("assistant"):
            with st.status("Running ActiveRAG Graph...", expanded=True) as status:
                try:
                    status.write("1. Preparing input...")
                    start_ts = time.time()
                    base_messages = convert_chat_to_base_messages(st.session_state.messages)

                    async def run_stream():
                        nonlocal state_updates, current_execution_flow
                        async for node_name, node_output in stream_active_rag(
                            query=user_query,
                            chat_history=base_messages,
                            input_urls=input_urls
                        ):
                            status.write(f"🔄 Executing `{node_name}`")
                            serial = to_serializable(node_output)
                            st.code(json.dumps(serial, indent=2))
                            current_execution_flow.append(node_name)
                            # merge each node’s outputs into our cumulative dict
                            state_updates.update(serial)

                    # run the async stream
                    asyncio.run(run_stream())
                    st.session_state.execution_flow = current_execution_flow

                    if st.session_state.get("last_feedback"):
                        state_updates["previous_feedback"] = st.session_state.last_feedback
                        # Optional: clear after use
                        st.session_state.last_feedback = None

                    # build final GraphState from merged dict
                    final_state = GraphState(**state_updates)
                    elapsed = time.time() - start_ts
                    status.success("✅ Graph execution complete!")
                    status.write(f" Completed in **{elapsed:.2f}** seconds")

                except Exception as e:
                    logger.exception(f"Error running ActiveRAG graph: {e}")
                    status.error(f"❌ Error during execution: {e}")
                    final_state = None

        # --- Display Results ---
        if final_state:
            display_state_details(final_state)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_state.answer or "Sorry, I couldn't generate a response."
            })
        else:
            error_message = "Sorry, an error occurred during processing."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

    # --- Feedback UI (Always visible if there's at least one assistant message) ---
    if any(m["role"] == "assistant" for m in st.session_state.messages):
        st.divider()
        with st.expander("🔍 Your Feedback", expanded=False):
            with st.form(key="feedback_form"):
                rating = st.slider(
                    "How would you rate this answer?",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="rating_slider"
                )
                comments = st.text_area(
                    "Any comments or suggestions?",
                    key="feedback_comments"
                )
                submitted = st.form_submit_button("Submit Feedback")

                if submitted:
                    st.session_state.last_feedback = {
                        "rating": rating,
                        "comments": comments.strip()
                    }
                    fb_msg = f"[Feedback: {rating} stars] {comments.strip()}"
                    st.session_state.messages.append({
                        "role": "system",
                        "content": fb_msg
                    })
                    st.success("Thanks! Your feedback will help refine future answers.")

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        # Load from .env file in the project root (adjust path if needed)
        dotenv_path = os.path.join(project_root, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            logger.info("Loaded environment variables from .env file.")
        else:
            logger.warning(".env file not found, relying on system environment variables.")
    except ImportError:
        logger.warning("'python-dotenv' not installed. Relying on system environment variables.")

    main()