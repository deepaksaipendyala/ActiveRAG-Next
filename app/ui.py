import streamlit as st
import sys
import os
os.environ["TORCH_DISABLE_STREAMLIT_WATCH"] = "1"
import asyncio
import logging
import json # For potentially displaying structured data nicely

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import graph runner and state (ensure backend is ready)
# We might need a modified runner that accepts config
from graph.builder import run_active_rag # Assuming run_rag handles state init
from graph.state import GraphState # Import the state model
from langchain_core.messages import HumanMessage, AIMessage

def convert_chat_to_base_messages(chat_list):
    messages = []
    for m in chat_list:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
        # If you later add system messages etc., handle them here
    return messages

# Setup basic logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - UI - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="ActiveRAG Next 🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def display_state_details(state: GraphState):
    """Helper to display various details from the final state in tabs."""
    if not state:
        st.warning("Graph execution did not return a final state.")
        return

    # Use tabs for better organization
    tab_answer, tab_reasoning, tab_context, tab_analysis, tab_interactive, tab_flow = st.tabs([
        "✅ Final Answer",
        "🧠 Reasoning",
        "📚 Context",
        "📊 Analysis",
        "💬 Interaction",
        "⚡ Execution Flow"
    ])

    # --- Tab 1: Final Answer ---
    with tab_answer:
        st.subheader("Final Answer")
        if state.answer:
            st.markdown(state.answer) # Display the final formatted answer
        else:
            st.warning("No final answer was generated.")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Validation Status", value=("✅ Valid" if state.is_valid else "⚠️ Invalid") if state.is_valid is not None else "N/A")
        with col2:
            st.metric(label="Confidence Score", value=f"{state.confidence_score:.2f}" if state.confidence_score is not None else "N/A")

        if state.validation_feedback and not state.is_valid:
             with st.expander("Show Validation Feedback"):
                  st.warning(state.validation_feedback)

    # --- Tab 2: Reasoning ---
    with tab_reasoning:
        st.subheader("Reasoner Output")
        if state.generation:
            st.text_area("Raw Generation Output", state.generation, height=200)
        else:
            st.info("No raw generation output available.")

        st.subheader("Reasoning Steps (GoT Trace)")
        if state.reasoning_steps:
            st.markdown("```\n" + "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(state.reasoning_steps)) + "\n```")
        else:
            st.info("No detailed reasoning steps available.")

        st.subheader("Final Context Used for Reasoning (Trace)")
        if state.reasoning_trace:
            st.text_area("Context Trace", state.reasoning_trace, height=300)
        else:
            st.info("No reasoning context trace available.")

    # --- Tab 3: Context ---
    with tab_context:
        st.subheader("Retrieved Content")
        if state.retrieved_contents:
            st.info(f"{len(state.retrieved_contents)} passages retrieved.")
            with st.expander("View All Retrieved Passages", expanded=False):
                for idx, passage in enumerate(state.retrieved_contents):
                    st.markdown(f"**Passage {idx+1}:**")
                    st.text_area(f"passage_{idx+1}", passage, height=150, label_visibility="collapsed")
                    st.markdown("---")
        else:
            st.warning("No content was retrieved.")
        # Add display for input URLs if implemented
        # if state.input_urls:
        #    st.subheader("User Provided URLs")
        #    st.write(state.input_urls)

    # --- Tab 4: Analysis ---
    with tab_analysis:
        st.subheader("Analyst Output")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entities Extracted", len(state.extracted_entities or []))
            if state.extracted_entities:
                 with st.expander("View Entities"):
                      st.write(state.extracted_entities)

        with col2:
            st.metric("Relations Extracted", len(state.extracted_relations or []))
            if state.extracted_relations:
                 with st.expander("View Relations (Triples)"):
                      st.json([rel.dict() for rel in state.extracted_relations]) # Display relations as JSON

        st.subheader("Simple Knowledge Graph")
        if state.knowledge_graph:
            # Pretty print the dictionary
            st.json(state.knowledge_graph)
        else:
            st.info("No knowledge graph generated.")

    # --- Tab 5: Interaction ---
    with tab_interactive:
        st.subheader("Interactive Agent Output")
        if state.clarification_questions:
            st.markdown("**Suggested Clarification Questions:**")
            for q in state.clarification_questions:
                st.markdown(f"- {q}")
        else:
            st.info("No clarification questions generated.")

        st.divider()

        if state.suggested_followups:
            st.markdown("**Suggested Follow-up Questions:**")
            for q in state.suggested_followups:
                st.markdown(f"- {q}")
        else:
            st.info("No follow-up questions generated.")

    # --- Tab 6: Execution Flow ---
    with tab_flow:
        st.subheader("Agent Execution Flow")
        # This requires capturing the flow during graph.stream()
        if 'execution_flow' in st.session_state and st.session_state.execution_flow:
            st.markdown(" -> ".join(st.session_state.execution_flow))
            # Optionally display state changes at each step if captured
        else:
            st.info("Execution flow tracking not implemented or graph run via invoke.")

        # Placeholder for graph visualization
        st.markdown("*(Graph visualization placeholder - Requires graphviz)*")
        # try:
        #     # Example: Generate and display the graph structure (static)
        #     from graph.builder import build_graph
        #     graph_viz = build_graph().get_graph()
        #     st.graphviz_chart(graph_viz.draw_mermaid_png()) # Requires pygraphviz/pydot
        # except Exception as viz_error:
        #     st.warning(f"Could not render graph visualization: {viz_error}")


# --- Main Application Logic ---
def main():
    st.title("🧠 ActiveRAG Next - Multi-Agent Reasoning Demo")
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
        st.subheader("Add Web References")
        input_urls_str = st.text_area(
            "Enter URLs (one per line):",
            height=100,
            placeholder="e.g., https://example.com/article\nhttps://another-site.org/info"
        )
        input_urls = [url.strip() for url in input_urls_str.split("\n") if url.strip()]

        # Document Set Selection (Placeholder)
        st.subheader("Select Document Context")
        # TODO: Implement logic to list and select available document sets/indices
        st.info("Document set selection not yet implemented.")
        # selected_docs = st.multiselect("Use Documents:", ["Set A", "Set B"], default=["Set A"]) # Example

        st.divider()
        run_button = st.button("🚀 Run ActiveRAG", use_container_width=True)

    # --- Main Area Logic ---
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
        # TODO: Modify run_rag or initial state creation to accept these configs
        rag_config = {
            "llm_provider": selected_llm, # Need to map this back to config values
            "input_urls": input_urls,
            # "selected_docs": selected_docs, # Pass selected docs if implemented
        }

        # --- Run the Graph ---
        final_state = None
        with st.chat_message("assistant"):
            status = st.status("🚀 Running ActiveRAG Graph...", expanded=False)
            with status:
                try:
                    st.write("1. Initializing graph...")
                    # ✨ FIX ✨
                    base_messages = convert_chat_to_base_messages(st.session_state.messages)
                    final_state = asyncio.run(run_active_rag(
                        query=user_query, chat_history=base_messages
                    ))
                    st.write("✅ Graph execution complete.")
                    status.update(label="✅ Graph execution complete!", state="complete", expanded=False)

                except Exception as e:
                    logger.exception(f"Error running ActiveRAG graph: {e}")
                    st.error(f"An error occurred: {e}")
                    status.update(label="❌ Error during execution", state="error", expanded=True)


            # --- Display Results ---
            if final_state:
                 # Store execution flow if captured (requires stream modification)
                 # st.session_state.execution_flow = capture_flow_from_stream(...)
                 display_state_details(final_state)

                 # Add final answer to chat history
                 assistant_response = final_state.answer if final_state.answer else "Sorry, I couldn't generate a response."
                 st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                 # Rerun to update chat display NOT needed if elements are displayed outside chat message context

            else:
                 st.error("Failed to get results from the RAG process.")
                 st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error occurred."})


if __name__ == "__main__":
    # TODO: Ensure necessary environment variables (API keys) are set via .env
    # Load environment variables (if using python-dotenv)
    # from dotenv import load_dotenv
    # load_dotenv()
    main()
