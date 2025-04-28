# prompts/coordinator.py

from langchain.prompts import ChatPromptTemplate

def build_coordinator_prompt(query: str, chat_history: list = None) -> ChatPromptTemplate:
    """
    Builds a simple prompt to classify the user's query.
    """
    # FIX: properly handle HumanMessage / AIMessage objects
    if chat_history:
        history = "\n".join(
            m.content if hasattr(m, "content") else str(m)
            for m in chat_history
        )
    else:
        history = "No prior history."

    template = (
        f"Given the following user query and chat history, classify the query into one of: "
        f"[knowledge, reasoning, critique].\n\n"
        f"Chat History:\n{history}\n\n"
        f"User Query:\n{query}\n\n"
        f"Respond with exactly one word."
    )
    return ChatPromptTemplate.from_template(template)

