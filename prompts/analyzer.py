# prompts/analyzer.py

from typing import List

ANALYZER_PROMPT = """
You are an expert information - extraction assistant.  
Your only output MUST be a single, valid JSON object with exactly two keys:

1. "entities": an array of all distinct entity names mentioned in the text passages.
2. "relations": an array of objects, each with exactly three string fields:
   • "subject"   - the entity performing an action or having an attribute  
   • "predicate" - the verb or relationship label  
   • "object"    - the target entity or attribute  

⚠️ NO Markdown, commentary, or extra keys—only the raw JSON.

Example output format:

{{
  "entities": ["Alice", "Bob", "Acme Corp"],
  "relations": [
    {{ "subject": "Alice", "predicate": "works_at",  "object": "Acme Corp" }},
    {{ "subject": "Bob",   "predicate": "manages",   "object": "Alice" }}
  ]
}}

Text passages:
{passages}
"""

def build_analyzer_prompt(passages: List[str]) -> str:
    """
    Build the prompt for the LLM to extract entities & relations as JSON.
    """
    # join with a clear separator so the model can see passage boundaries
    joined = "\n\n---\n\n".join(passages)
    return ANALYZER_PROMPT.format(passages=joined)
