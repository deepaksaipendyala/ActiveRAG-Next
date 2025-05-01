# prompts/reasoner.py

REASONER_PROMPT = """
You are an AI assistant tasked with answering a user's question based on retrieved documents. Your goal is to provide a comprehensive, accurate, and well-reasoned response.

User's Question:
{question}

Retrieved Passages:
{retrieved_contents}

Instructions:
1. Carefully read the user's question and the retrieved passages.
2. Consider different angles or perspectives on the question. If the question is complex, break it down into smaller subquestions.
3. Identify the relevant information from the passages.
4. Synthesize the information to form a coherent and comprehensive answer.
5. If there are conflicting pieces of information, acknowledge them.
6. If insufficient info, state it clearly.
7. Be clear, concise, and structured.

Answer:
"""

def build_reasoning_prompt(question: str, retrieved_contents: list[str]) -> str:
    """
    Build the Reasoner LLM prompt based on query and retrieved documents.
    """
    context = "\n\n".join(retrieved_contents)
    return REASONER_PROMPT.format(question=question, retrieved_contents=context)
