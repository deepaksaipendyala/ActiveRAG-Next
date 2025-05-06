# prompts/validator.py

from typing import List, Optional

VALIDATOR_PROMPT = """
You are an AI assistant whose job is to check whether a generated answer is fully supported by the retrieved evidence.

User's Question:
{question}

Generated Answer:
{generation}

Retrieved Passages:
{retrieved}

{feedback_section}Instructions:
- Only judge supportability against the *Retrieved Passages*.
- If every factual claim in the answer is directly supported by at least one passage and it fully addresses the question, respond with:  
  GOOD: <one‑sentence justification>.
- If the answer itself is correct **but** the retrieved passages contain minor inaccuracies or omissions (so support is imperfect), respond with:  
  GOOD: <one‑sentence justification>. Final answer was <Generated Answer given in prompt - summarize the answer in one‑sentence>.
- Otherwise (any hallucination, missing info, or contradiction), respond with:  
  BAD: <one‑sentence justification>. Final answer was <GGenerated Answer given in prompt - summarize the answer in one‑sentence>.
- Do **NOT** include anything else.

Examples:

Question: Who wrote “Pride and Prejudice”?  
Retrieved Passages:  
1. “Pride and Prejudice” was written by Jane Austen in 1813.  
Answer: “It was written by Jane Austen.”  
→ GOOD: supported by passage 1.

Question: Who invented the telephone?  
Retrieved Passages:  
1. Alexander Graham Bell patented the first practical telephone in 1876.  
Answer: “It was invented by Antonio Meucci.”  
→ BAD: Antonio Meucci is not mentioned in the passages. Final answer was It was invented by Antonio Meucci.

Question: What is the capital of France?  
Retrieved Passages:  
1. Paris is the capital of France and its largest city.  
2. France is in Western Europe and its currency is the euro. (contains irrelevant detail)  
Answer: “Paris.”  
→ Good: answer is correct; minor irrelevant details in passage 2 do not affect support. Final answer was Paris.

Now evaluate:
"""


def build_validation_prompt(
    question: str,
    retrieved_contents: List[str],
    generation: str,
    feedback: Optional[str] = None
) -> str:
    """
    Build a clear Validator prompt including retrieved passages and optional user feedback.
    """
    retrieved = "\n\n".join(retrieved_contents)
    # Only include feedback section if we have feedback to consider.
    if feedback:
        feedback_section = f"User Feedback: {feedback}\n\n"
    else:
        feedback_section = ""
    return VALIDATOR_PROMPT.format(
        question=question,
        retrieved=retrieved,
        generation=generation,
        feedback_section=feedback_section
    )
