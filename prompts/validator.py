# prompts/validator.py

VALIDATOR_PROMPT = """
You are an AI assistant tasked with validating the accuracy and coherence of a generated response based on the user's question and the retrieved passages.

User's Question:
{question}

Generated Response:
{generation}

Instructions:

1. Carefully read the user's question and the generated response.
2. Check if the response accurately addresses the user's question.
3. Look for hallucinations or unsupported claims.
4. Confirm if the answer is relevant, complete, and clear.

Output:
- If everything is correct, respond with "GOOD".
- If there are issues, respond with "BAD".

Respond strictly with GOOD or BAD first, optionally followed by a brief reason.
"""

def build_validation_prompt(question: str, generation: str) -> str:
    """
    Build the Validator LLM prompt based on the original question and generated answer.
    """
    return VALIDATOR_PROMPT.format(question=question, generation=generation)
